"""PPO agent — trajectory buffer, multi-head loss, update, checkpoint."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl_env import OBS_DIM
from rl_network import Flip7Network


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    active_head: str
    log_prob: float
    value: float
    reward: float
    done: bool


class TrajectoryBuffer:
    def __init__(self) -> None:
        self.transitions: List[Transition] = []

    def add(self, t: Transition) -> None:
        self.transitions.append(t)

    def clear(self) -> None:
        self.transitions.clear()

    def set_final_reward(self, reward: float) -> None:
        """Set reward and done on the last transition (call when episode ends)."""
        if self.transitions:
            self.transitions[-1].reward = reward
            self.transitions[-1].done = True


class PPOAgent:
    def __init__(
        self,
        network: Flip7Network,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = "cuda",
    ) -> None:
        self.network = network.to(device)
        self.device = device
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self._scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    def update(self, buffer: TrajectoryBuffer) -> Dict[str, float]:
        """Run multi-head PPO update. Returns dict with actor_loss, critic_loss, entropy, total_loss."""
        if not buffer.transitions:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}

        # Discounted returns (respect episode boundaries: do not bootstrap across done=True)
        trans = buffer.transitions
        rewards = np.array([t.reward for t in trans], dtype=np.float32)
        dones = np.array([t.done for t in trans], dtype=bool)
        G = np.zeros_like(rewards)
        G[-1] = rewards[-1]
        for t in range(len(trans) - 2, -1, -1):
            if dones[t]:
                # Terminal step of an episode: return is just reward, do not use G[t+1] (next episode)
                G[t] = rewards[t]
            else:
                G[t] = rewards[t] + self.gamma * G[t + 1]

        # Group by head
        by_head: Dict[str, List[int]] = {}
        for i, t in enumerate(trans):
            by_head.setdefault(t.active_head, []).append(i)

        obs_np = np.stack([t.obs for t in trans]).astype(np.float32)
        obs_t = torch.from_numpy(obs_np).to(self.device)
        old_log_probs = np.array([t.log_prob for t in trans], dtype=np.float32)
        old_log_probs_t = torch.from_numpy(old_log_probs).to(self.device)
        actions = np.array([t.action for t in trans], dtype=np.int64)
        actions_t = torch.from_numpy(actions).to(self.device)
        G_t = torch.from_numpy(G).to(self.device).unsqueeze(1)

        # Advantages per-head normalized
        values_old = np.array([t.value for t in trans], dtype=np.float32)
        A = G - values_old
        A_by_head: Dict[str, np.ndarray] = {}
        for h, idx in by_head.items():
            a = A[idx]
            a = (a - a.mean()) / (a.std() + 1e-8)
            A_by_head[h] = a

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.ppo_epochs):
            n_heads_used = 0
            self.optimizer.zero_grad(set_to_none=True)
            autocast_ctx = (
                torch.amp.autocast("cuda") if self.device == "cuda" else contextlib.nullcontext()
            )
            with autocast_ctx:
                values_all = torch.zeros(len(trans), device=self.device)
                actor_loss_sum = torch.tensor(0.0, device=self.device)
                entropy_sum = torch.tensor(0.0, device=self.device)
                for head, indices in by_head.items():
                    if not indices:
                        continue
                    idx_t = torch.tensor(indices, device=self.device, dtype=torch.long)
                    o = obs_t[idx_t]
                    a_old_lp = old_log_probs_t[idx_t]
                    a_act = actions_t[idx_t]
                    adv = torch.from_numpy(A_by_head[head]).to(self.device).float().unsqueeze(1)
                    logits, value = self.network(o, head)
                    values_all[idx_t] = value.squeeze(-1).float()
                    dist = Categorical(logits=logits)
                    new_log_prob = dist.log_prob(a_act)
                    entropy = dist.entropy().mean()
                    ratio = torch.exp(new_log_prob - a_old_lp)
                    surr1 = ratio * adv.squeeze(-1)
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv.squeeze(-1)
                    actor_loss_sum = actor_loss_sum - torch.min(surr1, surr2).sum()
                    entropy_sum = entropy_sum + entropy * len(indices)
                    n_heads_used += 1
                if n_heads_used == 0:
                    continue
                L_critic = ((values_all - G_t.squeeze(-1)) ** 2).mean()
                n_trans = len(trans)
                actor_loss_mean = actor_loss_sum / n_trans
                entropy_mean = entropy_sum / n_trans
                loss = actor_loss_mean + self.value_coef * L_critic - self.entropy_coef * entropy_mean
            total_actor_loss = actor_loss_mean.item()
            total_critic_loss = L_critic.item()
            total_entropy = entropy_mean.item()
            if self._scaler is not None:
                self._scaler.scale(loss).backward()
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

        return {
            "actor_loss": total_actor_loss,
            "critic_loss": total_critic_loss,
            "entropy": total_entropy,
            "total_loss": total_actor_loss + self.value_coef * total_critic_loss - self.entropy_coef * total_entropy,
        }

    def save(self, path: str, episode: int) -> None:
        torch.save(
            {
                "state_dict": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "episode": episode,
            },
            path,
        )

    def load(self, path: str) -> int:
        ck = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ck["state_dict"])
        if "optimizer" in ck:
            self.optimizer.load_state_dict(ck["optimizer"])
        return ck.get("episode", 0)
