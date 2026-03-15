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
    parser.add_argument("--opponent", type=str, default=None, help="Opponent strategy")
    parser.add_argument("--episodes", type=int, default=3_000_000)
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--episodes-per-worker", type=int, default=1, help="Episodes each worker runs per weight load (parallel only); amortizes sync")
    parser.add_argument("--snapshot-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=4000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.eval:
        run_eval(args.eval, device=args.device, opponent=args.opponent)
        return

    from rl_selfplay import run_training

    run_training(
        num_episodes=args.episodes,
        num_envs=args.envs,
        episodes_per_worker=args.episodes_per_worker,
        snapshot_interval=args.snapshot_interval,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        resume_path=args.resume,
        device=args.device,
    )


def run_eval(checkpoint_path: str, device: str = "cuda", opponent: str = None) -> None:
    """Run games: RL agent (deterministic) vs ComputerPlayer strategies; print win-rate table."""
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

    print("Strategy          | Dfeated | Avg Def | Wins    | Win %  ")
    print("-" * 45)

    if opponent is not None:
        strategies = [s for s in strategies if s[0] == opponent]

    for name, hit_stay in strategies:
        defeated_players = 0
        rl_wins = 0
        num_games = 100 if opponent is None else 1000
        for i in range(num_games):
            if i % (num_games/10) == 0 and i > 0 and opponent is not None:
                print(f"Game {i} of {num_games}")
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
            pct_wins = 100.0 * rl_wins / num_games
            for p in game._players[1:]:
                if p.get_total_score() < players[0].get_total_score():
                    defeated_players += 1
        avg_defeated_players = defeated_players / num_games
        print(f"{name:17} | {defeated_players:7} | {avg_defeated_players:3.3f} | {rl_wins:7} | {pct_wins:5.1f}%")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down cleanly.")
        # Optionally: flush logs, save a last checkpoint, etc.
        sys.exit(0)
