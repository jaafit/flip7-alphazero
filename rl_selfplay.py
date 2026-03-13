"""Self-play training loop — parallel workers, opponent pool, logging."""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import time
import traceback
from collections import deque
from multiprocessing import shared_memory
from typing import Any, Deque, List, Optional, Tuple

from rl_agent import PPOAgent, TrajectoryBuffer, Transition
from rl_env import Flip7Env
from rl_network import Flip7Network

# Win threshold to avoid fragile float equality (spec: reward 1.0 = beat all 3)
WIN_REWARD_THRESHOLD = 1.0 - 1e-6


def _worker_episode(
    agent_net: Flip7Network,
    frozen_net: Flip7Network,
    env: Flip7Env,
) -> Tuple[float, List[tuple]]:
    """Run one episode; return (episode_reward, list of (obs, action, active_head, log_prob, value, reward, done))."""
    buffer = TrajectoryBuffer()
    obs, info = env.reset()
    active_head = info["active_head"]
    legal_mask = info["legal_mask"]
    episode_reward = 0.0

    while True:
        action, log_prob, value = agent_net.select_action(
            obs, active_head, legal_mask, deterministic=False
        )
        buffer.add(
            Transition(
                obs=obs.copy(),
                action=action,
                active_head=active_head,
                log_prob=log_prob,
                value=value,
                reward=0.0,
                done=False,
            )
        )
        obs, reward, done, info = env.step(action, active_head)
        if done:
            buffer.set_final_reward(reward)
            episode_reward = reward
            break
        active_head = info["active_head"]
        legal_mask = info["legal_mask"]

    data = [
        (t.obs, t.action, t.active_head, t.log_prob, t.value, t.reward, t.done)
        for t in buffer.transitions
    ]
    return episode_reward, data


def _worker_loop(
    worker_id: int,
    weights_queue: mp.Queue,
    result_queue: mp.Queue,
    device: str,
    episodes_per_worker: int,
) -> None:
    """Worker process: create nets and env once; each message loads state_dict and runs episodes_per_worker episodes."""
    from rl_network import Flip7Network

    agent_net = Flip7Network().to(device)
    agent_net.eval()
    frozen_net = Flip7Network().to(device)
    for p in frozen_net.parameters():
        p.requires_grad = False
    env = Flip7Env(silent=True)
    env.set_opponent_network(frozen_net)

    while True:
        try:
            msg = weights_queue.get()
            if msg is None:
                break
            if isinstance(msg, tuple) and len(msg) == 5 and msg[0] == "shm":
                _, name_agent, size_agent, name_frozen, size_frozen = msg
                shm_agent = shared_memory.SharedMemory(name=name_agent)
                shm_frozen = shared_memory.SharedMemory(name=name_frozen)
                try:
                    agent_bytes = bytes(shm_agent.buf[:size_agent])
                    frozen_bytes = bytes(shm_frozen.buf[:size_frozen])
                    agent_sd = pickle.loads(agent_bytes)
                    frozen_sd = pickle.loads(frozen_bytes)
                finally:
                    shm_agent.close()
                    shm_frozen.close()
            else:
                agent_sd, frozen_sd = msg
            agent_net.load_state_dict(agent_sd)
            frozen_net.load_state_dict(frozen_sd)
            agent_net.to(device)
            frozen_net.to(device)
            batch_pairs: List[Tuple[float, List[tuple]]] = []
            for _ in range(episodes_per_worker):
                episode_reward, data = _worker_episode(agent_net, frozen_net, env)
                batch_pairs.append((episode_reward, data))
            result_queue.put((worker_id, batch_pairs))
        except Exception:
            traceback.print_exc()
            result_queue.put((worker_id, []))  # sentinel so main does not hang
            break


def run_training(
    num_episodes: int = 100_000,
    num_envs: int = 24,
    episodes_per_worker: int = 1,
    snapshot_interval: int = 500,
    log_interval: int = 100,
    checkpoint_interval: int = 2000,
    checkpoint_dir: str = "checkpoints/",
    resume_path: Optional[str] = None,
    device: str = "cuda",
) -> None:
    """Run self-play PPO training. Uses num_envs parallel workers when num_envs > 1. Each worker runs episodes_per_worker episodes per batch (amortizes weight sync)."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    network = Flip7Network()
    agent = PPOAgent(network, device=device)
    if resume_path:
        start_episode = agent.load(resume_path)
    else:
        start_episode = 0

    frozen_net = Flip7Network()
    frozen_net.load_state_dict(network.state_dict())
    frozen_net.cpu()
    for p in frozen_net.parameters():
        p.requires_grad = False

    reward_window: Deque[float] = deque(maxlen=200)
    actor_loss_window: Deque[float] = deque(maxlen=200)
    critic_loss_window: Deque[float] = deque(maxlen=200)
    entropy_window: Deque[float] = deque(maxlen=200)
    snapshot_count = 0
    start_time = time.time()

    if num_envs <= 1:
        _run_training_single_process(
            agent=agent,
            network=network,
            frozen_net=frozen_net,
            num_episodes=num_episodes,
            start_episode=start_episode,
            snapshot_interval=snapshot_interval,
            log_interval=log_interval,
            checkpoint_interval=checkpoint_interval,
            checkpoint_dir=checkpoint_dir,
            reward_window=reward_window,
            actor_loss_window=actor_loss_window,
            critic_loss_window=critic_loss_window,
            entropy_window=entropy_window,
            snapshot_count_ref=[snapshot_count],
            start_time=start_time,
            device=device,
        )
        return

    _run_training_parallel(
        agent=agent,
        network=network,
        frozen_net=frozen_net,
        num_episodes=num_episodes,
        num_envs=num_envs,
        start_episode=start_episode,
        snapshot_interval=snapshot_interval,
        log_interval=log_interval,
        checkpoint_interval=checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
        reward_window=reward_window,
        actor_loss_window=actor_loss_window,
        critic_loss_window=critic_loss_window,
        entropy_window=entropy_window,
        snapshot_count_ref=[snapshot_count],
        start_time=start_time,
        device=device,
        episodes_per_worker=episodes_per_worker,
    )


def _run_training_single_process(
    agent: PPOAgent,
    network: Flip7Network,
    frozen_net: Flip7Network,
    num_episodes: int,
    start_episode: int,
    snapshot_interval: int,
    log_interval: int,
    checkpoint_interval: int,
    checkpoint_dir: str,
    reward_window: Deque[float],
    actor_loss_window: Deque[float],
    critic_loss_window: Deque[float],
    entropy_window: Deque[float],
    snapshot_count_ref: List[int],
    start_time: float,
    device: str,
) -> None:
    """Single-process training: one env, learner fills buffer, step(action, active_head) only."""
    env = Flip7Env(silent=False)
    env.set_opponent_network(frozen_net)
    buffer = TrajectoryBuffer()

    for episode in range(start_episode, num_episodes):
        buffer.clear()
        obs, info = env.reset()
        active_head = info["active_head"]
        legal_mask = info["legal_mask"]

        while True:
            action, log_prob, value = network.select_action(
                obs, active_head, legal_mask, deterministic=False
            )
            buffer.add(
                Transition(
                    obs=obs,
                    action=action,
                    active_head=active_head,
                    log_prob=log_prob,
                    value=value,
                    reward=0.0,
                    done=False,
                )
            )
            obs, reward, done, info = env.step(action, active_head)
            if done:
                buffer.set_final_reward(reward)
                reward_window.append(reward)
                break
            active_head = info["active_head"]
            legal_mask = info["legal_mask"]

        metrics = agent.update(buffer)
        actor_loss_window.append(metrics["actor_loss"])
        critic_loss_window.append(metrics["critic_loss"])
        entropy_window.append(metrics["entropy"])

        if (episode + 1) % snapshot_interval == 0:
            frozen_net.load_state_dict(agent.network.state_dict())
            frozen_net.cpu()
            env.set_opponent_network(frozen_net)
            snapshot_count_ref[0] += 1

        if (episode + 1) % checkpoint_interval == 0:
            path = os.path.join(checkpoint_dir, f"ep_{episode + 1}.pt")
            agent.save(path, episode + 1)

        if (episode + 1) % log_interval == 0:
            _log_progress(
                episode + 1,
                num_episodes,
                start_episode,
                start_time,
                reward_window,
                actor_loss_window,
                critic_loss_window,
                entropy_window,
                snapshot_count_ref[0],
            )


def _run_training_parallel(
    agent: PPOAgent,
    network: Flip7Network,
    frozen_net: Flip7Network,
    num_episodes: int,
    num_envs: int,
    start_episode: int,
    snapshot_interval: int,
    log_interval: int,
    checkpoint_interval: int,
    checkpoint_dir: str,
    reward_window: Deque[float],
    actor_loss_window: Deque[float],
    critic_loss_window: Deque[float],
    entropy_window: Deque[float],
    snapshot_count_ref: List[int],
    start_time: float,
    device: str,
    episodes_per_worker: int = 1,
) -> None:
    """Multi-process training: num_envs workers, each runs episodes_per_worker episodes per batch, learner merges and updates."""
    weights_queues: List[mp.Queue] = [mp.Queue() for _ in range(num_envs)]
    result_queue: mp.Queue = mp.Queue()
    episodes_per_batch = num_envs * episodes_per_worker

    workers = []
    for i in range(num_envs):
        p = mp.Process(
            target=_worker_loop,
            args=(i, weights_queues[i], result_queue, device, episodes_per_worker),
            daemon=True,
        )
        p.start()
        workers.append(p)

    try:
        episode = start_episode
        while episode < num_episodes:
            agent_sd = {k: v.cpu().clone() for k, v in agent.network.state_dict().items()}
            frozen_sd = {k: v.cpu().clone() for k, v in frozen_net.state_dict().items()}
            agent_bytes = pickle.dumps(agent_sd)
            frozen_bytes = pickle.dumps(frozen_sd)
            shm_agent = shared_memory.SharedMemory(create=True, size=len(agent_bytes))
            shm_frozen = shared_memory.SharedMemory(create=True, size=len(frozen_bytes))
            try:
                shm_agent.buf[:len(agent_bytes)] = agent_bytes
                shm_frozen.buf[:len(frozen_bytes)] = frozen_bytes
                msg = ("shm", shm_agent.name, len(agent_bytes), shm_frozen.name, len(frozen_bytes))
                for i in range(num_envs):
                    weights_queues[i].put(msg)
            finally:
                pass  # unlink after batch below

            batch_results: List[Optional[Tuple[int, List[Tuple[float, List[tuple]]]]]] = [None] * num_envs
            for _ in range(num_envs):
                wid, batch_pairs = result_queue.get()
                batch_results[wid] = (wid, batch_pairs)
                if not batch_pairs or any(not d for _, d in batch_pairs):
                    raise RuntimeError(
                        f"Worker {wid} crashed or returned empty episode; aborting training."
                    )

            shm_agent.close()
            shm_agent.unlink()
            shm_frozen.close()
            shm_frozen.unlink()

            merged = TrajectoryBuffer()
            for _wid, pairs in batch_results:
                for episode_reward, data in pairs:
                    reward_window.append(episode_reward)
                    for tup in data:
                        obs, action, active_head, log_prob, value, reward, done = tup
                        merged.add(
                            Transition(
                                obs=obs,
                                action=action,
                                active_head=active_head,
                                log_prob=log_prob,
                                value=value,
                                reward=reward,
                                done=done,
                            )
                        )

            metrics = agent.update(merged)
            actor_loss_window.append(metrics["actor_loss"])
            critic_loss_window.append(metrics["critic_loss"])
            entropy_window.append(metrics["entropy"])

            episode += episodes_per_batch
            batch_start = episode - episodes_per_batch

            # Snapshot when we crossed a snapshot_interval boundary
            if (episode - start_episode) // snapshot_interval > (batch_start - start_episode) // snapshot_interval:
                frozen_net.load_state_dict(agent.network.state_dict())
                frozen_net.cpu()
                snapshot_count_ref[0] += 1

            if (episode - start_episode) // checkpoint_interval > (batch_start - start_episode) // checkpoint_interval:
                path = os.path.join(checkpoint_dir, f"ep_{episode}.pt")
                agent.save(path, episode)

            if (episode - start_episode) // log_interval > (batch_start - start_episode) // log_interval:
                _log_progress(
                    episode,
                    num_episodes,
                    start_episode,
                    start_time,
                    reward_window,
                    actor_loss_window,
                    critic_loss_window,
                    entropy_window,
                    snapshot_count_ref[0],
                )
    finally:
        for q in weights_queues:
            q.put(None)
        for p in workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def _log_progress(
    episode: int,
    num_episodes: int,
    start_episode: int,
    start_time: float,
    reward_window: Deque[float],
    actor_loss_window: Deque[float],
    critic_loss_window: Deque[float],
    entropy_window: Deque[float],
    snapshot_count: int,
) -> None:
    elapsed = time.time() - start_time
    eps_per_sec = (episode - start_episode) / elapsed if elapsed > 0 else 0
    mean_reward = sum(reward_window) / len(reward_window) if reward_window else 0
    wins = sum(1 for r in reward_window if r >= WIN_REWARD_THRESHOLD)
    win_rate = wins / len(reward_window) if reward_window else 0
    mean_beaten = mean_reward * 3
    mean_actor = sum(actor_loss_window) / len(actor_loss_window) if actor_loss_window else 0
    mean_critic = sum(critic_loss_window) / len(critic_loss_window) if critic_loss_window else 0
    mean_entropy = sum(entropy_window) / len(entropy_window) if entropy_window else 0
    print(
        f"Ep {episode}/{num_episodes} | "
        f"eps/s {eps_per_sec:.1f} | "
        f"reward(avg) {mean_reward:.3f} | "
        f"win_rate {win_rate:.2%} | "
        f"beaten {mean_beaten:.2f} | "
        f"actor {mean_actor:.4f} critic {mean_critic:.4f} ent {mean_entropy:.4f} | "
        f"snap {snapshot_count}"
    )
