"""Entry point for Flip 7 RL training and evaluation."""

from __future__ import annotations

import argparse
import sys

from rl_agent import PPOAgent
from rl_network import Flip7Network


def main() -> None:
    parser = argparse.ArgumentParser(description="Flip 7 RL training")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--eval", type=str, default=None, help="Run evaluation with checkpoint path")
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--snapshot-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=2000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.eval:
        run_eval(args.eval, device=args.device, num_games=args.episodes)
        return

    from rl_selfplay import run_training

    run_training(
        num_episodes=args.episodes,
        num_envs=args.envs,
        snapshot_interval=args.snapshot_interval,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        resume_path=args.resume,
        device=args.device,
    )


def run_eval(checkpoint_path: str, device: str = "cuda", num_games: int = 1000) -> None:
    """Run 1000 games: RL agent (deterministic) vs ComputerPlayer strategies; print win-rate table."""
    from game import Game
    from rl_env import Flip7Env
    from computer_player import (
        ComputerPlayer,
        play_round_to,
        play_to_bust_probability,
        always_hit_strategy,
        random_hit_or_stay_strategy,
        adaptive_bust_probability_strategy,
        expected_value_strategy,
        hybrid_strategy,
        gap_based_strategy,
        optimal_strategy,
        target_leader_strategy,
        target_last_place_strategy,
    )

    network = Flip7Network()
    agent = PPOAgent(network, device=device)
    agent.load(checkpoint_path)
    network = agent.network
    network.eval()

    strategies = [
        ("play_20", play_round_to(20)),
        ("play_25", play_round_to(25)),
        ("play_30", play_round_to(30)),
        ("play_35", play_round_to(35)),
        ("bust_p0.2", play_to_bust_probability(0.2)),
        ("bust_p0.3", play_to_bust_probability(0.3)),
        ("always_hit", always_hit_strategy),
        ("random", random_hit_or_stay_strategy),
        ("adaptive_0.3", adaptive_bust_probability_strategy(0.3)),
        ("expected_value", expected_value_strategy),
        ("hybrid", hybrid_strategy),
        ("gap_based", gap_based_strategy),
        ("optimal", optimal_strategy),
    ]
    action_tgt = target_leader_strategy
    positive_tgt = target_last_place_strategy

    print("Strategy          | RL Wins | Total | Win %")
    print("-" * 45)

    for name, hit_stay in strategies:
        rl_wins = 0
        for _ in range(num_games):
            game = Game()
            game.set_silent_mode(True)
            from rl_agent_player import RLPlayer

            env = Flip7Env(silent=True)
            env.set_opponent_network(network)
            players = [
                RLPlayer("RL", network, 0, env, is_training_agent=False),
                ComputerPlayer(f"CPU_{name}", hit_stay, action_tgt, positive_tgt),
                ComputerPlayer(f"CPU_{name}b", hit_stay, action_tgt, positive_tgt),
                ComputerPlayer(f"CPU_{name}c", hit_stay, action_tgt, positive_tgt),
            ]
            env._game = game
            env._agent_player = players[0]
            game._players = players
            game._deck = __import__("deck").Deck()
            game._round = 1
            game._dealer_idx = 0
            while not game._has_winner():
                game._play_round()
                game._show_scores()
                game._next_round()
            winner = max(game._players, key=lambda p: p.get_total_score())
            if winner is players[0]:
                rl_wins += 1
        pct = 100.0 * rl_wins / num_games
        print(f"{name:17} | {rl_wins:7} | {num_games:5} | {pct:5.1f}%")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down cleanly.")
        # Optionally: flush logs, save a last checkpoint, etc.
        sys.exit(0)
