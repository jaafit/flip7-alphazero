"""Environment wrapper for Flip 7 RL — observations, step, reward."""

from __future__ import annotations

import queue
import threading
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import random
from card import Card, CardType, ActionType, ModifierType
from deck import Deck
from game import Game
from player import BasePlayer, PlayerState

N_PLAYERS = 4

# Observation dimension: 10*4 player + 6 deck + 1 meta (flip3 bust prob removed; use num_cnt, sc, pbust as proxies)
PLAYER_BLOCK_DIM = 11
DECK_BLOCK_DIM = 6
META_DIM = 3
OBS_DIM = PLAYER_BLOCK_DIM * N_PLAYERS + DECK_BLOCK_DIM + META_DIM



def format_obs(obs: np.ndarray) -> str:
    """Return obs as readable string with labels; values rounded to nearest hundredth."""
    n = len(obs)
    player_part_len = n - DECK_BLOCK_DIM - META_DIM
    if player_part_len <= 0 or player_part_len % PLAYER_BLOCK_DIM != 0:
        return str(obs.round(2))
    n_players = player_part_len // PLAYER_BLOCK_DIM
    player_labels = [
        "num_sum", "num_cnt", "x2", "mods", "sc", "active",
        "pbust", "round", "owed", "kept", "bags",
    ]
    parts: List[str] = []
    for i in range(n_players):
        start = i * PLAYER_BLOCK_DIM
        block = obs[start : start + PLAYER_BLOCK_DIM]
        pairs = ", ".join(f"{k}={round(float(v), 2)}" for k, v in zip(player_labels, block))
        parts.append(f"Player{i + 1}: {pairs}")
    deck_start = n_players * PLAYER_BLOCK_DIM
    deck_labels = ["p_x2", "p_f3", "p_fz", "p_sc", "ev", "ncards"]
    deck_block = obs[deck_start : deck_start + DECK_BLOCK_DIM]
    deck_str = ", ".join(f"{k}={round(float(v), 2)}" for k, v in zip(deck_labels, deck_block))
    parts.append(f"Deck: {deck_str}")

    meta_labels = ["dealer_norm", "solo", "is_last_round"]
    if META_DIM and deck_start + DECK_BLOCK_DIM < n:
        meta_block = obs[deck_start + DECK_BLOCK_DIM : deck_start + DECK_BLOCK_DIM + META_DIM]
        pairs = ", ".join(f"{k}={round(float(v), 2)}" for k, v in zip(meta_labels, meta_block))
        parts.append(f"meta: {pairs}")
    return ". ".join(parts)



def _player_block(p: BasePlayer, deck: Deck, bags: int) -> np.ndarray:
    """Encode one player. Flip3 bust not in obs; agent can use num_cnt, sc, pbust as proxies for target choice."""
    number_cards = [c for c in p.get_hand() if c.type == CardType.NUMBER]
    number_card_value = float(sum(c.value for c in number_cards))
    has_x2 = 1.0 if any(
        c.type == CardType.MODIFIER and c.modifier == ModifierType.MULTIPLY_2
        for c in p.get_hand()
    ) else 0.0
    plus_mod_total = sum(
        c.get_points() for c in p.get_hand()
        if c.type == CardType.MODIFIER and c.modifier != ModifierType.MULTIPLY_2
    )
    has_second_chance = 1.0 if p.has_second_chance() else 0.0
    is_active = 1.0 if p.is_active() else 0.0
    number_card_count = float(len(number_cards))
    # Round score max: (12+11+10+9+8+7)*2 + (2+4+6+8+10) + 15 = 171 (theoretical; rarely reached)
    round_score = p.calculate_round_score() 
    potential_score = (p.get_total_score() + round_score)
    total_score = p.get_total_score() + (round_score if not p.is_active() else 0)

    p_bust_hit = _compute_bust_prob_if_hit(p, deck)

    out = np.zeros(PLAYER_BLOCK_DIM, dtype=np.float32)
    out[0] = number_card_value / (12+11+10+9+8+7)
    out[1] = number_card_count / 6.0
    out[2] = has_x2
    out[3] = plus_mod_total / 30.0
    out[4] = has_second_chance
    out[5] = is_active
    out[6] = p_bust_hit
    out[7] = round_score / 171.0
    out[8] = potential_score / 300.0
    out[9] = total_score / 300.0
    out[10] = bags / (N_PLAYERS - 1)
    return out


def _deck_block(deck: Deck) -> np.ndarray:
    """Encode deck features. Uses current_player for bust probs."""
    draw_cards = deck.cards
    n = len(draw_cards)
    out = np.zeros(DECK_BLOCK_DIM, dtype=np.float32)
    if n == 0:
        draw_cards = deck.discards()
        n = len(draw_cards)
    if n == 0:
        return out  # no cards to sample; leave features at 0
    number_sum = sum(c.value for c in draw_cards if c.type == CardType.NUMBER)
    plus_mod_sum = sum(c.get_points() for c in draw_cards if c.type == CardType.MODIFIER) # x2 get_points() is 0
    excpected_draw_value = (number_sum + plus_mod_sum) / n
    
    out[0] = sum(1 for c in draw_cards if c.type == CardType.MODIFIER and c.modifier == ModifierType.MULTIPLY_2) / n 
    out[1] = sum(1 for c in draw_cards if c.type == CardType.ACTION and c.action == ActionType.FLIP_THREE) / n
    out[2] = sum(1 for c in draw_cards if c.type == CardType.ACTION and c.action == ActionType.FREEZE) / n
    out[3] = sum(1 for c in draw_cards if c.type == CardType.ACTION and c.action == ActionType.SECOND_CHANCE) / n
    out[4] = excpected_draw_value / 12.0
    out[5] = n / 94.0
    return out


def _compute_bust_prob_if_hit(player: BasePlayer, deck: Deck) -> float:
    """Probability of busting if current player draws one (number) card. Flip3 branch omitted for speed."""
    draw_cards = deck._cards if len(deck._cards) > 0 else deck._discards
    if not draw_cards or player.has_second_chance():
        return 0.0
    number_vals = {c.value for c in player.get_hand() if c.type == CardType.NUMBER}
    bust_count = sum(1 for c in draw_cards if c.type == CardType.NUMBER and c.value in number_vals)
    return bust_count / len(draw_cards)


class Flip7Env:
    """Environment wrapper: game loop runs in thread; step/reset via queues."""

    def __init__(self, agent_player_idx: int = 0, silent: bool = True) -> None:
        self.agent_player_idx = agent_player_idx
        self.silent = silent
        self._game: Optional[Game] = None
        self._agent_player: Optional[BasePlayer] = None
        self._obs_queue: queue.Queue = queue.Queue(maxsize=1)
        self._action_queue: queue.Queue = queue.Queue(maxsize=1)
        self._current_active_head: Optional[str] = None
        self._current_legal_mask: Optional[np.ndarray] = None
        self._trajectory_buffer: Optional[Any] = None
        self._opponent_network: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None
        self._done_event = threading.Event()
        self._game_ready = threading.Event()

    def put_obs_and_wait_action(
        self, obs: np.ndarray, active_head: str, legal_mask: np.ndarray
    ) -> Tuple[int, str]:
        """Called by RLPlayer from game thread: put (obs, ...); block until (action, active_head) received.
        Returns (action, active_head) only. The trajectory buffer is filled by the training loop that
        calls step(action, active_head)—that loop has the network and adds transitions with log_prob/value."""
        self._obs_queue.put((obs, active_head, legal_mask, 0.0, False))
        return self._action_queue.get()  # (action, active_head)

    def set_opponent_network(self, net: Any) -> None:
        self._opponent_network = net

    def set_trajectory_buffer(self, buf: Any) -> None:
        self._trajectory_buffer = buf

    def has_active_game(self) -> bool:
        """True if a game is running (external game set via set_game) or agent player is set."""
        return self._game is not None

    def set_game(self, game: Optional[Game]) -> None:
        """Set external game for interactive play (encode_state/get_legal_mask use this game)."""
        self._game = game

    def encode_state(self, player_index: int) -> np.ndarray:
        """Build obs float32 array from current game state (agent perspective)."""
        if self._game is None:
            return np.zeros(OBS_DIM, dtype=np.float32)
        game = self._game
        players = game._players
        n = len(players)
        relative_dealer_idx = (game._dealer_idx - player_index) % n
        ordered = [players[(player_index + i) % n] for i in range(n)]
        potential_scores = [p.get_total_score() + p.calculate_round_score() for p in ordered]
        blocks = []
        solo = 0
        for i, p in enumerate(ordered):
            if p.is_active() and not any(p2.is_active() for p2 in ordered if p2 is not p):
                solo = 1
            bags = 0
            p_potential_score = potential_scores[i]
            for j in range(n):
                if j != i:
                    if potential_scores[j] < p_potential_score:
                        bags += 1
            blocks.append(_player_block(p, game._deck, bags))
        player_part = np.concatenate(blocks)
        deck_part = _deck_block(game._deck)
        is_last_round = 1 if any(score > 200 and not ordered[i].is_active() for i, score in enumerate(potential_scores)) else 0
        meta = np.array([
            relative_dealer_idx / n,
            solo,
            is_last_round,
        ], dtype=np.float32)
        return np.concatenate([player_part, deck_part, meta])

    def get_legal_mask(self, active_head: str, player_index: Optional[int] = None) -> np.ndarray:
        """Return boolean numpy array for the given head, in the requesting player's observation space.
        For n-player heads, mask[i] = True means the player in observation block i is a valid target."""
        if self._game is None:
            if active_head == "hit_stay":
                return np.ones(2, dtype=bool)
            return np.zeros(N_PLAYERS, dtype=bool)
        if player_index is None:
            player_index = self.agent_player_idx
        game = self._game
        players = game._players
        n = len(players)
        current = players[player_index] if player_index < n else None

        if active_head == "hit_stay":
            mask = np.zeros(2, dtype=bool)
            if current is not None and current.is_active():
                mask[1] = True   # HIT
                if current.has_cards():
                    mask[0] = True  # STAY
            return mask

        if active_head == "freeze" or active_head == "flip3":
            game_mask = np.zeros(n, dtype=bool)
            for i in range(n):
                if players[i].is_active():
                    game_mask[i] = True
            return np.array(
                [game_mask[(player_index + j) % n] for j in range(n)],
                dtype=bool,
            )

        if active_head == "second_chance":
            game_mask = np.zeros(n, dtype=bool)
            for i in range(n):
                if i == player_index:
                    continue
                p = players[i]
                if p.is_active() and not p.has_second_chance():
                    game_mask[i] = True
            return np.array(
                [game_mask[(player_index + j) % n] for j in range(n)],
                dtype=bool,
            )

        return np.zeros(n, dtype=bool)

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Construct new Game with N_PLAYERS RLPlayers, run until first agent decision; return (obs, info)."""
        from rl_agent_player import RLPlayer
        from rl_network import Flip7Network
        # Create a minimal network for placeholder if needed; env will set real one
        dummy_net = Flip7Network()
        self._obs_queue = queue.Queue(maxsize=1)
        self._action_queue = queue.Queue(maxsize=1)
        self._done_event.clear()
        self._game_ready.clear()
        self._current_active_head = None
        self._current_legal_mask = None

        def run_game() -> None:
            try:
                game = Game()
                game.set_silent_mode(True)
                agent_idx = self.agent_player_idx
                opponents_net = self._opponent_network or dummy_net
                players: List[BasePlayer] = []
                for i in range(N_PLAYERS):
                    is_agent = i == agent_idx
                    net = dummy_net if is_agent else opponents_net
                    name = "RLAgent" if is_agent else f"RLOpp{i}"
                    pl = RLPlayer(name, net, i, self, is_agent)
                    players.append(pl)
                game._players = players
                game._deck = Deck()
                game._round = 1
                game._dealer_idx = random.randint(0, N_PLAYERS - 1)
                self._game = game
                self._agent_player = players[agent_idx]
                self._game_ready.set()
                while not game._has_winner():
                    game._play_round()
                    game._next_round()
                agent_score = self._agent_player.get_total_score()
                opponents = [p for p in game._players if p is not self._agent_player]
                opponents_beaten = sum(1 for o in opponents if agent_score > o.get_total_score())
                reward = opponents_beaten / (N_PLAYERS - 1)
                self._obs_queue.put((None, None, None, reward, True))
            except Exception as e:
                import traceback
                traceback.print_exc()
                try:
                    self._obs_queue.put((None, None, None, 0.0, True))
                except Exception:
                    pass
            finally:
                self._done_event.set()

        self._thread = threading.Thread(target=run_game, daemon=True)
        self._thread.start()
        self._game_ready.wait()
        # Wait for first decision from game thread (RLPlayer puts obs and blocks)
        obs, active_head, legal_mask, reward, done = self._obs_queue.get()
        self._current_active_head = active_head or "hit_stay"
        self._current_legal_mask = legal_mask if legal_mask is not None else np.ones(2, dtype=bool)
        info = {"active_head": self._current_active_head, "legal_mask": self._current_legal_mask}
        if done:
            return self.encode_state(self.agent_player_idx), info
        return obs, info

    def step(
        self, action: int, active_head: str
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Feed (action, active_head) to game thread; return (obs, reward, done, info). Spec: 2 args only."""
        self._action_queue.put((action, active_head))
        obs, active_head, legal_mask, reward, done = self._obs_queue.get()
        self._current_active_head = active_head
        self._current_legal_mask = legal_mask
        if done:
            return self.encode_state(self.agent_player_idx), reward, True, {"active_head": active_head, "legal_mask": legal_mask}
        return obs, 0.0, False, {"active_head": active_head, "legal_mask": legal_mask}
