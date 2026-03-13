# RL training bottleneck analysis (--envs scaling)

With `--envs 1` you get ~10 episodes/sec; with `--envs 12` only ~16 episodes/sec. So 12× envs give only ~1.6× throughput. Here’s what’s shared and where the bottleneck is.

## What all workers share (and where contention is)

- **One main process**  
  - Owns the learner: `PPOAgent` / `Flip7Network` and `frozen_net`.  
  - Only the main process does `agent.update(...)` and optimizer steps.

- **Per-batch flow (parallel path in `rl_selfplay._run_training_parallel`)**
  1. Main builds `agent_sd` and `frozen_sd` (full `state_dict()` copied to CPU).
  2. Main **puts the same `(agent_sd, frozen_sd)` into each of `num_envs` queues** (e.g. 12 times).
  3. Each worker **gets** from its own queue, **load_state_dict** twice, runs **one full episode** (env in a thread, many `select_action` + `env.step`), then **puts** `(worker_id, episode_reward, trajectory_data)` into a **single shared `result_queue`**.
  4. Main **get()s from `result_queue`** exactly `num_envs` times (once per worker), merges trajectories, then runs **one** `agent.update(merged)` for the whole batch.

So “what they all share” is:

- The **same weight payload** (agent + frozen state dicts) sent from main to every worker each batch.
- A **single result queue** that all workers put to and main drains.
- A **barrier**: the main loop does not start the next batch until **all** workers have finished and been collected.

## Bottlenecks (in rough impact order)

### 1. Weight broadcast: 12× serialization of the same state dicts

- **Where:** `rl_selfplay.py` lines 288–291: main does  
  `agent_sd = {k: v.cpu().clone() for ...}`, `frozen_sd = ...`, then  
  `for i in range(num_envs): weights_queues[i].put((agent_sd, frozen_sd))`.
- **Why it hurts:** `mp.Queue.put` **pickles** the payload. So the **same** ~(agent_sd, frozen_sd) is pickled **once per worker** (e.g. 12 times) every batch. That’s a lot of CPU and memory bandwidth on the main process, and the workers then each unpickle one full copy.
- **Fix (implemented):** Serialize once and broadcast the bytes via **shared memory** (`multiprocessing.shared_memory`). Main: `pickle.dumps` once per batch, write to two `SharedMemory` segments, send only `("shm", name_agent, size_agent, name_frozen, size_frozen)` to each worker. Workers: attach by name, read bytes, `pickle.loads`, then `load_state_dict`. Main unlinks the segments after collecting all results. This removes 11 of the 12 pickles on the main process (see `rl_selfplay._run_training_parallel` and `_worker_loop`).

### 2. Barrier: batch time = slowest worker

- **Where:** Main loop blocks with `for _ in range(num_envs): result_queue.get()`.
- **Why it hurts:** Episode length varies. When you have 12 envs, the batch finishes when the **slowest** of the 12 episodes finishes. Other workers may sit idle after finishing their episode, and main does nothing until all 12 results are in.
- **Possible fixes:**  
  - **Async / streaming:** Let main consume results as they arrive and either update more often (e.g. with smaller batches) or keep a rolling buffer. This changes the algorithm (e.g. toward async PPO) and needs care.  
  - **Multiple episodes per worker (implemented):** Each worker runs `--episodes-per-worker` episodes per weight load and returns that many trajectories; fewer barrier syncs per episode. Try e.g. `--envs 12 --episodes-per-worker 2` or `3`.  
  - **Multiple envs per worker:** Alternatively, fewer worker processes each running several envs (same idea). You then have fewer workers, so fewer weight broadcasts and less barrier penalty, while keeping total env count high.  
  - **Non-blocking batch:** Start the next batch of weight sends as soon as **some** (e.g. N − 1) results are in and reuse the last worker’s slot; this requires a more involved protocol.

### 3. Result transfer: 12 large trajectory payloads

- **Where:** Each worker does `result_queue.put((worker_id, episode_reward, data))` with `data = list of (obs, action, active_head, log_prob, value, reward, done)`; obs is length-89 float32.
- **Why it hurts:** Every episode can be dozens of steps. So you’re pickling 12 large lists of mixed types (numpy arrays, strings, floats, bools) each batch. One shared queue also means all workers contend on the same pipe/socket.
- **Possible fixes:**  
  - Prefer **numeric arrays** (e.g. `np.stack` obs, actions, etc.) and send one array per field to reduce pickle overhead and size.  
  - **Shared memory for trajectories:** Main preallocates or workers write into shared buffers; main only receives small “ready” tokens and then reads from shared memory. More involved.

### 4. Single main thread: serialize → wait → merge → update

- Main does all weight serialization and all result merging and the single PPO update. So main is busy at the start of the batch (serialization), then idle (waiting for workers), then busy again (merge + update). The 12× pickle in step 1 keeps that “busy” phase long and makes the barrier wait start later.

### 5. Inference on CPU in workers (and why GPU wasn't used)

- Workers created `Flip7Network()` and never received a `device` argument, so they never called `.to("cuda")` and **select_action** ran on CPU. With 12 processes, that’s 12 CPU-bound inference threads (plus game threads). - **Why it was done this way:** (1) Avoid multiprocessing + CUDA in child processes (spawn + CUDA is supported but each process gets its own GPU context and model copy, which can OOM or add overhead). (2) Simplicity: one GPU user (the main process for `agent.update()`) and workers stayed CPU-only.
- **Using GPU in workers:** The training script now passes `device` into the worker; both agent and frozen nets are moved to that device. With one GPU and 12 workers, that's 12 processes each holding a copy of the model on the same GPU — memory use is 12× model size (usually fine for a small policy). All 12 run inference on the same GPU, which can still be faster than 12× CPU inference.

## Summary

- **Shared:** One learner on main; same weight payload broadcast to every worker; one result queue; a strict barrier so the next batch only starts after all workers finish.
- **Largest bottleneck:** Main process **pickling the same (agent_sd, frozen_sd) once per worker** every batch. Fixing this with a single serialization + shared-memory broadcast should reduce main-side CPU and improve scaling.
- **Next:** Barrier (batch = slowest worker) and large trajectory result payloads. Improving both would likely require protocol/algorithm changes (async or multi-env-per-worker and/or more compact result format).
