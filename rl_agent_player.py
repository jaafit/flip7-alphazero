"""RL agent player for Flip 7 — BasePlayer subclass, single integration point with game."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from card import ActionType
from player import BasePlayer, GameState

from rl_env import format_obs

# hit_stay: index 0=STAY, 1=HIT (agent picks the action with the *higher* logit)
HIT_STAY_LABELS = ("STAY", "HIT")


def _format_logits(active_head: str, logits: Any, action: int) -> str:
    """Format logits with labels and show which action was chosen (higher logit = chosen)."""
    logits = logits.round(2)
    if active_head == "hit_stay" and len(logits) == 2:
        pairs = ", ".join(f"{HIT_STAY_LABELS[i]}={logits[i]}" for i in range(2))
        chose = HIT_STAY_LABELS[action]
        return f"{pairs} (chose {chose})"
    # freeze, flip3, second_chance: logits are per player index in obs order
    n = len(logits)
    pairs = ", ".join(f"P{i}={logits[i]}" for i in range(n))
    return f"{pairs} (chose P{action})"


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
        show_obs_and_head: bool = False,
    ) -> None:
        super().__init__(name)
        self.init(name)
        self._network = network
        self._player_idx = player_idx
        self._env_ref = env_ref
        self._is_training_agent = is_training_agent
        self._show_obs_and_head = show_obs_and_head

    def get_player_icon(self) -> str:
        return "🧠"

    def _select_action_from_network(
        self, obs: Any, active_head: str, legal_mask: Any
    ) -> int:
        """Call network for action (inference only); optionally print obs/logits."""
        with torch.no_grad():
            out = self._network.select_action(
                obs, active_head, legal_mask, deterministic=False, return_logits=self._show_obs_and_head
            )
        action = int(out[0])
        if self._show_obs_and_head:
            logits = out[3]
            hand_str = ", ".join(str(c) for c in self.get_hand()) or "empty"
            print(f"[Agent {self.get_name()}] hand: {hand_str}")
            print(f"[Agent {self.get_name()}] obs: {format_obs(obs)}")
            print(f"[Agent {self.get_name()}] head={active_head} logits: {_format_logits(active_head, logits, action)}")
        return action

    def make_hit_stay_decision(self, game_state: GameState) -> bool:
        assert self._env_ref.has_active_game(), "RLPlayer decision called before env has an active game"
        obs = self._env_ref.encode_state(self._player_idx)
        active_head = "hit_stay"
        legal_mask = self._env_ref.get_legal_mask(active_head, self._player_idx)
        if self._is_training_agent:
            action, _ = self._env_ref.put_obs_and_wait_action(obs, active_head, legal_mask)  # _ is active_head; buffer filled by loop that calls step()
            return action == 1  # 0=STAY, 1=HIT
        action = self._select_action_from_network(obs, active_head, legal_mask)
        return action == 1

    def _action_to_game_player(self, action: int, game_state: GameState) -> BasePlayer:
        """Map network action (observation-space index) to game player."""
        n = len(game_state.players)
        game_idx = (self._player_idx + action) % n
        return game_state.players[game_idx]

    def choose_action_target(
        self, game_state: GameState, action_type: ActionType
    ) -> BasePlayer:
        assert self._env_ref.has_active_game(), "RLPlayer decision called before env has an active game"
        obs = self._env_ref.encode_state(self._player_idx)
        active_head = "freeze" if action_type == ActionType.FREEZE else "flip3"
        legal_mask = self._env_ref.get_legal_mask(active_head, self._player_idx)
        if self._is_training_agent:
            action, _ = self._env_ref.put_obs_and_wait_action(obs, active_head, legal_mask)  # _ is active_head
            return self._action_to_game_player(action, game_state)
        action = self._select_action_from_network(obs, active_head, legal_mask)
        return self._action_to_game_player(action, game_state)

    def choose_positive_action_target(
        self, game_state: GameState, action_type: ActionType
    ) -> BasePlayer:
        assert self._env_ref.has_active_game(), "RLPlayer decision called before env has an active game"
        active_head = "second_chance"
        legal_mask = self._env_ref.get_legal_mask(active_head, self._player_idx)
        if not legal_mask.any():
            raise RuntimeError("No legal Second Chance target")
        obs = self._env_ref.encode_state(self._player_idx)
        if self._is_training_agent:
            action, _ = self._env_ref.put_obs_and_wait_action(obs, active_head, legal_mask)  # _ is active_head
            return self._action_to_game_player(action, game_state)
        action = self._select_action_from_network(obs, active_head, legal_mask)
        return self._action_to_game_player(action, game_state)
