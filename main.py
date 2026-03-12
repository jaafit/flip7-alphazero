#!/usr/bin/env python3
"""Entry point for Flip 7 CLI."""

import argparse
import sys

from game import Game


def main() -> None:
    parser = argparse.ArgumentParser(description="Flip 7 - Press your luck to 200 points!")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to manually choose cards")
    args = parser.parse_args()

    print("🎴 Welcome to Flip 7!")
    print("Press your luck and flip your way to 200 points!")
    if args.debug:
        print("🐛 DEBUG MODE: You can choose cards manually!")
    print()

    game = Game()
    game.set_debug_mode(args.debug)
    try:
        game.run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
