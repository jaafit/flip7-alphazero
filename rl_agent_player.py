"""RL agent player for Flip 7 — BasePlayer subclass, single integration point with game."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from card import ActionType
from player import BasePlayer, GameState

if TYPE_CHECKING:
    from rl_env import Flip7Env


class RLPlayer(BasePlayer):
    """Player controlled by the RL network; uses env_ref for queue and buffer."""

    def __init__(
        self,
        name: str,
        network: Any,
        player_idx: int,
        env_ref: "Flip7Env",
        is_training_agent: bool,
    ) -> None:
        super().__init__(name)
        self.init(name)
        self._network = network
        self._player_idx = player_idx
        self._env_ref = env_ref
        self._is_training_agent = is_training_agent

    def get_player_icon(self) -> str:
        return "🧠"

    def make_hit_stay_decision(self, game_state: GameState) -> bool:
        assert self._env_ref.has_active_game(), "RLPlayer decision called before env has an active game"
        obs = self._env_ref.encode_state()
        active_head = "hit_stay"
        legal_mask = self._env_ref.get_legal_mask(active_head)
        if self._is_training_agent:
            action, _ = self._env_ref.put_obs_and_wait_action(obs, active_head, legal_mask)  # _ is active_head; buffer filled by loop that calls step()
            return action == 1  # 0=STAY, 1=HIT
        with torch.no_grad():
            action, _, _ = self._network.select_action(
                obs, active_head, legal_mask, deterministic=False
            )
        return action == 1

    def choose_action_target(
        self, game_state: GameState, action_type: ActionType
    ) -> BasePlayer:
        assert self._env_ref.has_active_game(), "RLPlayer decision called before env has an active game"
        obs = self._env_ref.encode_state()
        active_head = "freeze" if action_type == ActionType.FREEZE else "flip3"
        legal_mask = self._env_ref.get_legal_mask(active_head)
        if self._is_training_agent:
            action, _ = self._env_ref.put_obs_and_wait_action(obs, active_head, legal_mask)  # _ is active_head
            return game_state.players[action]
        with torch.no_grad():
            action, _, _ = self._network.select_action(
                obs, active_head, legal_mask, deterministic=False
            )
        return game_state.players[action]

    def choose_positive_action_target(
        self, game_state: GameState, action_type: ActionType
    ) -> BasePlayer:
        assert self._env_ref.has_active_game(), "RLPlayer decision called before env has an active game"
        active_head = "second_chance"
        legal_mask = self._env_ref.get_legal_mask(active_head)
        if not legal_mask.any():
            raise RuntimeError("No legal Second Chance target")
        obs = self._env_ref.encode_state()
        if self._is_training_agent:
            action, _ = self._env_ref.put_obs_and_wait_action(obs, active_head, legal_mask)  # _ is active_head
            return game_state.players[action]
        with torch.no_grad():
            action, _, _ = self._network.select_action(
                obs, active_head, legal_mask, deterministic=False
            )
        return game_state.players[action]
