"""Sanity checks for Flip 7 RL (spec Section 12)."""

import sys
import numpy as np
import torch

from deck import Deck
from rl_env import OBS_DIM, Flip7Env, _compute_bust_prob_if_hit
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
    hit_stay = info["legal_mask"]
    assert hit_stay.shape == (2,)
    assert np.any(hit_stay), "hit_stay mask should have at least one True"
    for head in ("freeze", "flip3", "second_chance"):
        m = env.get_legal_mask(head)
        assert m.shape == (4,)
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
    heads = [("hit_stay", 2), ("freeze", 4), ("flip3", 4), ("second_chance", 4)]
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
