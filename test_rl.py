"""Sanity checks for Flip 7 RL (spec Section 12)."""

import copy
import math
import random
import sys
import numpy as np
import torch

from card import Card, CardType, ActionType
from deck import Deck
from player import BasePlayer
from rl_env import OBS_DIM, N_PLAYERS, Flip7Env, _compute_bust_prob_if_hit, _compute_flip3_bust_prob
from rl_network import Flip7Network
from rl_agent import PPOAgent, TrajectoryBuffer, Transition


def test_obs_shape_and_finiteness():
    env = Flip7Env(silent=True)
    obs, _ = env.reset()
    assert obs.dtype == np.float32, obs.dtype
    assert obs.shape == (OBS_DIM,), obs.shape
    assert np.all(np.isfinite(obs)), f"non-finite obs: {obs}"
    print("test_obs_shape_and_finiteness OK")



def test_deck_size():
    deck = Deck()
    assert len(deck.cards) + len(deck.discards) == 94, f"deck total should be 94, got {len(deck.cards) + len(deck.discards)}"
    print("test_deck_size OK")


def test_legal_mask_at_start():
    env = Flip7Env(silent=True)
    _, info = env.reset()
    active_head = info["active_head"]
    legal_mask = info["legal_mask"]
    expected_shape = (2,) if active_head == "hit_stay" else (N_PLAYERS,)
    assert legal_mask.shape == expected_shape, (
        f"legal_mask shape for active_head={active_head}: expected {expected_shape}, got {legal_mask.shape}"
    )
    assert np.any(legal_mask), "legal mask should have at least one True"
    for head in ("freeze", "flip3", "second_chance"):
        m = env.get_legal_mask(head)
        assert m.shape == (N_PLAYERS,)
    print("test_legal_mask_at_start OK")


def test_step_completes():
    env = Flip7Env(silent=True)
    obs, info = env.reset()
    obs, reward, done, info = env.step(1, "hit_stay")
    assert obs.shape == (OBS_DIM,), obs.shape
    assert obs.dtype == np.float32
    print("test_step_completes OK")


def test_full_episode():
    env = Flip7Env(silent=True)
    obs, info = env.reset()
    active_head = info["active_head"]
    legal_mask = info["legal_mask"]
    steps = 0
    while True:
        action = int(np.argmax(legal_mask)) if np.any(legal_mask) else 0
        obs, reward, done, info = env.step(action, active_head)
        steps += 1
        if done:
            assert reward in (0.0, 1/3, 2/3, 1.0), reward
            assert np.all(np.isfinite(obs))
            break
        active_head = info["active_head"]
        legal_mask = info["legal_mask"]
        if steps > 2000:
            raise RuntimeError("episode too long")
    print("test_full_episode OK")


def test_network_forward():
    net = Flip7Network()
    x = torch.zeros(1, OBS_DIM)
    logits, value = net.forward(x, "hit_stay")
    assert logits.shape == (1, 2), logits.shape
    assert value.shape == (1, 1), value.shape
    print("test_network_forward OK")


def test_all_heads():
    net = Flip7Network()
    x = torch.zeros(1, OBS_DIM)
    heads = [("hit_stay", 2), ("freeze", N_PLAYERS), ("flip3", N_PLAYERS), ("second_chance", N_PLAYERS)]
    for active_head, size in heads:
        logits, value = net.forward(x, active_head)
        assert logits.shape == (1, size), (active_head, logits.shape)
        assert value.shape == (1, 1)
    print("test_all_heads OK")


def test_ppo_update():
    env = Flip7Env(silent=True)
    buffer = TrajectoryBuffer()
    obs, info = env.reset()
    active_head = info["active_head"]
    legal_mask = info["legal_mask"]
    net = Flip7Network()
    agent = PPOAgent(net, device="cpu")
    for _ in range(5):
        action, log_prob, value = net.select_action(obs, active_head, legal_mask, deterministic=False)
        buffer.add(Transition(obs=obs, action=action, active_head=active_head, log_prob=log_prob, value=value, reward=0.0, done=False))
        obs, reward, done, info = env.step(action, active_head)
        if done:
            buffer.set_final_reward(reward)
            break
        active_head = info["active_head"]
        legal_mask = info["legal_mask"]
    if buffer.transitions:
        metrics = agent.update(buffer)
        for k, v in metrics.items():
            assert np.isfinite(v), f"{k}={v}"
    print("test_ppo_update OK")


def test_bust_probability():
    from card import Card, CardType
    from player import BasePlayer

    player = BasePlayer("t")
    player.add_card(Card.new_number_card(5))
    deck_cards = [
        Card.new_number_card(5),
        Card.new_number_card(3),
        Card.new_number_card(7),
    ]
    deck = Deck()
    deck._cards = deck_cards
    p = _compute_bust_prob_if_hit(player, deck, is_only_active_player=True)
    assert 0 <= p <= 1, p
    print("test_bust_probability OK")


def test_opponent_swap():
    env = Flip7Env(silent=True)
    net1 = Flip7Network()
    net2 = Flip7Network()
    env.set_opponent_network(net1)
    obs, info = env.reset()
    env.set_opponent_network(net2)
    obs, _, _, _ = env.step(1, "hit_stay")
    assert obs.shape == (OBS_DIM,)
    print("test_opponent_swap OK")


def test_checkpoint_roundtrip():
    net = Flip7Network()
    agent = PPOAgent(net, device="cpu")
    agent.save("/tmp/test_rl_ckpt.pt", episode=42)
    net2 = Flip7Network()
    agent2 = PPOAgent(net2, device="cpu")
    ep = agent2.load("/tmp/test_rl_ckpt.pt")
    assert ep == 42
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    mask = np.ones(2, dtype=bool)
    a1, _, _ = agent.network.select_action(obs, "hit_stay", mask, deterministic=True)
    a2, _, _ = agent2.network.select_action(obs, "hit_stay", mask, deterministic=True)
    assert a1 == a2, (a1, a2)
    print("test_checkpoint_roundtrip OK")


def _simulate_flip3_trials(deck_list, discard_list, number_vals, has_second_chance, n_trials=1000, seed=None):
    """Run n_trials flip-3 simulations; return empirical bust proportion.
    Each trial: shuffle deck and discards, draw 3 in order (deck first, then reshuffle discards when empty).
    Process each card in order: a number that duplicates current hand causes bust (or uses second chance);
    a new number is added to hand so drawing that value again later would bust.
    If seed is not None, RNG is reset at the start (for reproducibility)."""
    if seed is not None:
        random.seed(seed)
    busts = 0
    for _ in range(n_trials):
        deck = copy.copy(deck_list)
        discards = copy.copy(discard_list)
        random.shuffle(deck)
        random.shuffle(discards)
        drawn = []
        for _ in range(3):
            if not deck:
                deck = discards
                discards = []
                random.shuffle(deck)
            drawn.append(deck.pop())
        # Process in order: duplicate = bust (or second chance); new number grows hand
        hand_vals = set(number_vals)
        second_chance_left = 1 if has_second_chance else 0
        busted = False
        for c in drawn:
            if c.type != CardType.NUMBER:
                continue
            if c.value in hand_vals:
                if second_chance_left > 0:
                    second_chance_left -= 1
                else:
                    busted = True
                    break
            else:
                hand_vals = hand_vals | {c.value}
        if busted:
            busts += 1
    return busts / n_trials


def test_flip3_bust_prob_without_second_chance():
    """Empirical flip-3 bust rate (no second chance) should match _compute_flip3_bust_prob at 99% confidence."""
    # Hand has a 5 → any number card with value 5 in the 3 draws is a buster
    deck_list = [
        Card.new_number_card(3),
        Card.new_number_card(2),
    ]
    discard_list = [
        Card.new_number_card(9),
        Card.new_number_card(7),
        Card.new_number_card(5),
        Card.new_number_card(4),
        Card.new_number_card(6),
    ]
    number_vals = {5}
    has_second_chance = False

    player = BasePlayer("t")
    player.add_card(Card.new_number_card(5))
    deck = Deck()
    deck._cards = list(deck_list)
    deck._discards = list(discard_list)

    p_pred = _compute_flip3_bust_prob(player, True, deck)
    n_trials = 10000
    n_seeds = 5
    # Use multiple seeds so a single unlucky seed doesn't fail the test
    p_hats = [
        _simulate_flip3_trials(deck_list, discard_list, number_vals, has_second_chance, n_trials=n_trials, seed=s)
        for s in range(n_seeds)
    ]
    p_hat = sum(p_hats) / n_seeds
    # Margin for the mean of n_seeds runs of n_trials each
    n_effective = n_trials * n_seeds
    margin = 2.576 * math.sqrt(p_pred * (1 - p_pred) / n_effective)
    if abs(p_hat - p_pred) > margin:
        raise AssertionError(
            f"Flip-3 bust prob (no second chance): predicted={p_pred:.4f}, empirical mean={p_hat:.4f} "
            f"(n_seeds={n_seeds}, n_trials={n_trials}); 99%% CI margin={margin:.4f}"
        )
    print(f"test_flip3_bust_prob_without_second_chance {p_pred:.4f} OK")


def test_flip3_bust_prob_with_second_chance():
    """Empirical flip-3 bust rate (with second chance) should match _compute_flip3_bust_prob at 99% confidence."""
    deck_list = [
        Card.new_number_card(5),
        Card.new_number_card(3),
        Card.new_number_card(7),
        Card.new_number_card(3),
        Card.new_number_card(9),
        Card.new_number_card(10),
        Card.new_number_card(11),
    ]
    discard_list = [
        Card.new_number_card(5),
        Card.new_number_card(4),
        Card.new_number_card(5),
        Card.new_number_card(1),
    ]
    number_vals = {5}
    has_second_chance = True

    player = BasePlayer("t")
    player.add_card(Card.new_number_card(5))
    player.add_card(Card.new_action_card(ActionType.SECOND_CHANCE))
    deck = Deck()
    deck._cards = list(deck_list)
    deck._discards = list(discard_list)

    p_pred = _compute_flip3_bust_prob(player, True, deck)
    n_trials = 10000
    n_seeds = 5
    p_hats = [
        _simulate_flip3_trials(deck_list, discard_list, number_vals, has_second_chance, n_trials=n_trials, seed=s)
        for s in range(n_seeds)
    ]
    p_hat = sum(p_hats) / n_seeds
    n_effective = n_trials * n_seeds
    margin = 2.576 * math.sqrt(p_pred * (1 - p_pred) / n_effective)
    if abs(p_hat - p_pred) > margin:
        raise AssertionError(
            f"Flip-3 bust prob (with second chance): predicted={p_pred:.4f}, empirical mean={p_hat:.4f} "
            f"(n_seeds={n_seeds}, n_trials={n_trials}); 99%% CI margin={margin:.4f}"
        )
    print(f"test_flip3_bust_prob_with_second_chance {p_pred:.4f} OK")


def main():
    tests = [
        test_obs_shape_and_finiteness,
        test_deck_size,
        test_legal_mask_at_start,
        test_step_completes,
        test_full_episode,
        test_network_forward,
        test_all_heads,
        test_ppo_update,
        test_bust_probability,
        test_flip3_bust_prob_without_second_chance,
        test_flip3_bust_prob_with_second_chance,
        test_opponent_swap,
        test_checkpoint_roundtrip,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}", file=sys.stderr)
            raise
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
