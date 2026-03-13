"""Environment wrapper for Flip 7 RL — observations, step, reward."""

from __future__ import annotations

import queue
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import random
from card import Card, CardType, ActionType, ModifierType
from deck import Deck
from game import Game
from player import BasePlayer, PlayerState

# Observation dimension: 19*4 player + 11 deck + 2 meta = 89 (spec Section 13)
PLAYER_BLOCK_DIM = 11
DECK_BLOCK_DIM = 6
META_DIM = 1
OBS_DIM = PLAYER_BLOCK_DIM * 4 + DECK_BLOCK_DIM + META_DIM 
N_PLAYERS = 4


def _player_block(p: BasePlayer, deck: Deck, is_only_active_player: bool) -> np.ndarray:
    """Encode one player."""
    number_cards = [c for c in p.get_hand() if c.type == CardType.NUMBER]
    number_card_value = float(sum(c.value for c in number_cards))
    has_x2 = 1.0 if any(
        c.type == CardType.MODIFIER and c.modifier == ModifierType.MULTIPLY_2
        for c in p.get_hand()
    ) else 0.0
    plus_mod_total = sum(
        c.get_points() for c in p.get_hand()
        if c.type == CardType.MODIFIER and c.modifier != ModifierType.MULTIPLY_2
    ) / 30.0
    has_second_chance = 1.0 if p.has_second_chance() else 0.0
    is_active = 1.0 if p.is_active() else 0.0
    number_card_count = float(len(number_cards))
    round_score = p.calculate_round_score() / 300.0
    total_score = p.get_total_score() / 300.0

    p_bust_hit = _compute_bust_prob_if_hit(p, deck, is_only_active_player)
    p_bust_flip3 = _compute_flip3_bust_prob(p, deck) 
    

    out = np.zeros(PLAYER_BLOCK_DIM, dtype=np.float32)
    out[0] = number_card_value
    out[1] = has_x2
    out[2] = plus_mod_total
    out[3] = has_second_chance
    out[4] = is_active
    out[5] = 1 if is_only_active_player else 0
    out[6] = number_card_count
    out[7] = p_bust_hit
    out[8] = p_bust_flip3
    out[9] = round_score
    out[10] = total_score
    return out


def _deck_block(deck: Deck) -> np.ndarray:
    """Encode deck features. Uses current_player for bust probs."""
    draw_cards = deck.cards
    n = len(draw_cards)
    out = np.zeros(DECK_BLOCK_DIM, dtype=np.float32)
    if n == 0: 
        draw_cards = deck.discards().shuffle()
        n = len(draw_cards)
    
    number_sum = sum(c.value for c in draw_cards if c.type == CardType.NUMBER)
    plus_mod_sum = sum(c.get_points() for c in draw_cards if c.type == CardType.MODIFIER) # x2 get_points() is 0
    excpected_draw_value = (number_sum + plus_mod_sum) / n / 12.0
    
    out[0] = sum(1 for c in draw_cards if c.type == CardType.MODIFIER and c.modifier == ModifierType.MULTIPLY_2) / n 
    out[1] = sum(1 for c in draw_cards if c.type == CardType.ACTION and c.action == ActionType.FLIP_THREE) / n
    out[2] = sum(1 for c in draw_cards if c.type == CardType.ACTION and c.action == ActionType.FREEZE) / n
    out[3] = sum(1 for c in draw_cards if c.type == CardType.ACTION and c.action == ActionType.SECOND_CHANCE) / n
    out[4] = excpected_draw_value
    out[5] = n / 94.0
    return out


def _compute_bust_prob_if_hit(player: BasePlayer, deck:Deck, is_only_active_player: bool) -> float:
    """Probability of busting if current player draws one card."""
    draw_cards = deck._cards if len(deck._cards) > 0 else deck._discards
    number_vals = {c.value for c in player.get_hand() if c.type == CardType.NUMBER}
    bust_count = sum(1 for c in draw_cards if c.type == CardType.NUMBER and c.value in number_vals)
    p_number_bust = bust_count / len(draw_cards)
    p_draw_flip3 = sum(1 for c in draw_cards if c.type == CardType.ACTION and c.action == ActionType.FLIP_THREE) / len(draw_cards)
    p_flip3_bust = _compute_flip3_bust_prob(player, deck) if is_only_active_player and p_draw_flip3 else 0.0 # todo: subtract drawn flip3
    return 1 - (1-p_number_bust) * (1-p_flip3_bust * p_draw_flip3)


def _compute_flip3_bust_prob(player: BasePlayer, deck: Deck) -> float:
    """P(bust) if forced to draw exactly 3 cards sequentially (no replacement)."""
    # todo: not yet accounting for possibility of drawing another flip3 card and being sole active player
    number_vals = set(c.value for c in player.get_hand() if c.type == CardType.NUMBER)
    busters_in_deck = sum(1 for c in deck.cards if c.type == CardType.NUMBER and c.value in number_vals)
    busters_in_discards = sum(1 for c in deck._discards if c.type == CardType.NUMBER and c.value in number_vals)
    deck_size = len(deck.cards)
    discards_size = len(deck._discards)
    safe_prob = 1.0
    for i in range(3):
        if deck_size:
            safe_prob *= (deck_size - busters_in_deck) / deck_size
            deck_size -= 1
        else:
            safe_prob *= (discards_size - busters_in_discards) / discards_size
            discards_size -= 1
    return 1.0 - safe_prob


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
        """True if a game is running and agent player is set (safe to call encode_state)."""
        return self._game is not None and self._agent_player is not None

    def encode_state(self, player_index:int) -> np.ndarray:
        """Build OBS_DIM float32 array from current game state (agent perspective)."""
        if self._game is None or self._agent_player is None:
            return np.zeros(OBS_DIM, dtype=np.float32)
        game = self._game
        players = game._players
        n = len(players)
        ordered: List[BasePlayer] = []
        for i in range(n):
            idx = (player_index + i) % n
            ordered.append(players[idx])
        blocks = []
        for p in ordered:
            is_only_active_player = p.is_active() and not any(p2.is_active() for p2 in ordered if p2 is not p)
            blocks.append(_player_block(p, game._deck, is_only_active_player))
        player_part = np.concatenate(blocks)
        deck_part = _deck_block(game._deck)
        meta = np.array([
            game._dealer_idx / N_PLAYERS,
        ], dtype=np.float32)
        return np.concatenate([player_part, deck_part, meta])

    def get_legal_mask(self, active_head: str) -> np.ndarray:
        """Return boolean numpy array for the given head."""
        if self._game is None or self._agent_player is None:
            if active_head == "hit_stay":
                return np.ones(2, dtype=bool)
            return np.zeros(N_PLAYERS, dtype=bool)
        game = self._game
        agent = self._agent_player
        players = game._players

        if active_head == "hit_stay":
            mask = np.zeros(2, dtype=bool)
            if agent.is_active():
                mask[1] = True   # HIT
                if agent.has_cards():
                    mask[0] = True  # STAY
            return mask

        if active_head == "freeze" or active_head == "flip3":
            mask = np.zeros(N_PLAYERS, dtype=bool)
            for i in range(N_PLAYERS):
                if i < len(players) and players[i].is_active():
                    mask[i] = True
            return mask

        if active_head == "second_chance":
            mask = np.zeros(N_PLAYERS, dtype=bool)
            for i in range(N_PLAYERS):
                if i >= len(players):
                    continue
                p = players[i]
                if p is agent:
                    continue
                if p.is_active() and not p.has_second_chance():
                    mask[i] = True
            return mask

        return np.zeros(N_PLAYERS, dtype=bool)

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
