"""Human player implementation for Flip 7."""

from typing import Callable, Optional

from card import ActionType
from player import BasePlayer, GameState


class HumanPlayer(BasePlayer):
    """Human player that reads hit/stay and targets from stdin."""

    def __init__(self, name: str, readline: Optional[Callable[[], str]] = None) -> None:
        super().__init__(name)
        self.init(name)
        self._readline = readline or (lambda: input())

    def get_player_icon(self) -> str:
        return "👤"

    def make_hit_stay_decision(self, game_state: GameState) -> bool:
        print(f"{self.name}'s hand, {[str(c) for c in self.get_hand()]}")
        print(f"🎯 {self.name}, do you want to (H)it or (S)tay? ", end="", flush=True)
        while True:
            try:
                choice = self._readline().strip().lower()
            except Exception:
                raise RuntimeError("failed to read input")
            if choice in ("h", "hit"):
                return True
            if choice in ("s", "stay"):
                return False
            print("Please enter 'H' for Hit or 'S' for Stay: ", end="", flush=True)

    def choose_action_target(
        self, game_state: GameState, action_type: ActionType
    ) -> BasePlayer:
        action_names = {
            ActionType.FREEZE: "Who should be frozen?",
            ActionType.FLIP_THREE: "Who should flip three cards?",
            ActionType.SECOND_CHANCE: "Who should get the Second Chance card?",
        }
        print(f"   {action_names[action_type]}")
        for i, p in enumerate(game_state.active_players, 1):
            print(f"   {i}) {p.get_name()}")
        n = len(game_state.active_players)
        while True:
            print(f"Enter choice (1-{n}): ", end="", flush=True)
            try:
                line = self._readline().strip()
            except Exception:
                raise RuntimeError("failed to read input")
            try:
                choice = int(line)
            except ValueError:
                print(f"Please enter a number between 1 and {n}: ", end="", flush=True)
                continue
            if 1 <= choice <= n:
                return game_state.active_players[choice - 1]
            print(f"Please enter a number between 1 and {n}: ", end="", flush=True)

    def choose_positive_action_target(
        self, game_state: GameState, action_type: ActionType
    ) -> BasePlayer:
        return self.choose_action_target(game_state, action_type)
