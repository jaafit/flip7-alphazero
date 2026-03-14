"""PyTorch network for Flip 7 RL — shared encoder, actor heads, critic."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl_env import OBS_DIM, N_PLAYERS


def load_network_from_checkpoint(path: str, device: str = "cpu") -> Tuple["Flip7Network", int]:
    """Load checkpoint and return (network, n_players). obs_dim and n_players inferred from state_dict."""
    ck = torch.load(path, map_location=device)
    state_dict = ck.get("state_dict", ck)
    obs_dim = int(state_dict["encoder.0.weight"].shape[1])
    n_players = int(state_dict["actor_freeze.weight"].shape[0])
    net = Flip7Network(obs_dim=obs_dim, n_players=n_players).to(device)
    net.load_state_dict(state_dict)
    net.eval()
    return net, n_players


class Flip7Network(nn.Module):
    """Shared encoder + 4 actor heads (hit_stay, freeze, flip3, second_chance) + critic."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_players: int = N_PLAYERS,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_players = n_players
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.actor_hit_stay = nn.Linear(hidden_dim, 2)
        self.actor_freeze = nn.Linear(hidden_dim, n_players)
        self.actor_flip3 = nn.Linear(hidden_dim, n_players)
        self.actor_second_chance = nn.Linear(hidden_dim, n_players)
        self.critic = nn.Linear(hidden_dim, 1)

    def _head_logits(self, h: torch.Tensor, active_head: str) -> torch.Tensor:
        if active_head == "hit_stay":
            return self.actor_hit_stay(h)
        if active_head == "freeze":
            return self.actor_freeze(h)
        if active_head == "flip3":
            return self.actor_flip3(h)
        if active_head == "second_chance":
            return self.actor_second_chance(h)
        raise ValueError(f"unknown active_head: {active_head}")

    def forward(
        self, obs: torch.Tensor, active_head: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits for active head, value)."""
        h = self.encoder(obs)
        logits = self._head_logits(h, active_head)
        value = self.critic(h)
        return logits, value

    def select_action(
        self,
        obs: np.ndarray,
        active_head: str,
        legal_mask: np.ndarray,
        deterministic: bool = False,
        return_logits: bool = False,
    ) -> Tuple[int, float, float, ...]:
        """Returns (action_int, log_prob_float, value_float) or + (logits_np,) if return_logits."""
        with torch.no_grad():
            if obs.ndim == 1:
                obs = obs[np.newaxis, :]
            obs = obs.astype(np.float32) if obs.dtype != np.float32 else obs
            x = torch.from_numpy(obs).to(next(self.parameters()).device)
            logits, value = self.forward(x, active_head)
            mask_t = torch.from_numpy(legal_mask).to(device=logits.device)
            # Broadcast mask to logits shape (e.g. [2] -> [1, 2] when obs was unsqueezed)
            if mask_t.shape != logits.shape:
                mask_t = mask_t.broadcast_to(logits.shape)
            logits[~mask_t] = -1e9
            dist = Categorical(logits=logits)
            if deterministic:
                action_t = torch.argmax(logits, dim=-1)
                log_prob = 0.0  # unused during eval
            else:
                action_t = dist.sample()
                log_prob = dist.log_prob(action_t).item()
            action = action_t.item()
            val = value.squeeze(-1).item()
            if return_logits:
                return action, log_prob, val, logits.squeeze(0).cpu().numpy()
            return action, log_prob, val
