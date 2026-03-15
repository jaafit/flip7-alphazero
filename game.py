"""Main game logic and flow for Flip 7."""

import random
import sys
from typing import List, Optional, Callable

from card import Card, ActionType
from deck import Deck
from player import BasePlayer, GameState
from human_player import HumanPlayer
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
    target_random_strategy,
)

COMPUTER_NAMES = [
    "HAL", "Data", "GLaDOS", "WALL-E", "EVE", "R2D2", "C3PO", "T-800",
    "Skynet", "Optimus", "Megatron", "Bender", "WOPR", "Cortana", "Marvin",
]


class Game:
    """Main game state and loop."""

    def __init__(self, readline: Optional[Callable[[], str]] = None) -> None:
        self._players: List[BasePlayer] = []
        self._deck = Deck()
        self._round = 1
        self._dealer_idx = 0
        self._readline = readline or (lambda: input())
        self._debug_mode = False
        self._silent_mode = False
        self._agent_network = None
        self._agent_n_players: Optional[int] = None
        self._agent_env = None

    def set_debug_mode(self, debug: bool) -> None:
        self._debug_mode = debug
        self._deck.set_debug_mode(debug, self._readline)

    def set_silent_mode(self, silent: bool) -> None:
        self._silent_mode = silent

    def set_agent_checkpoint(self, checkpoint_path: str) -> None:
        """Load RL checkpoint; player count will be inferred from network in _setup_players."""
        from rl_network import load_network_from_checkpoint
        self._agent_network, self._agent_n_players = load_network_from_checkpoint(checkpoint_path)

    def _printf(self, fmt: str, *args: object) -> None:
        if not self._silent_mode:
            print(fmt % args, end="", flush=True)

    def _println(self, *args: object) -> None:
        if not self._silent_mode:
            print(*args)

    def _print(self, *args: object) -> None:
        if not self._silent_mode:
            print(*args, end="", flush=True)

    def run(self) -> None:
        self._setup_players()
        if self._agent_env is not None:
            self._agent_env.set_game(self)
        self._println("\n🎮 Starting Flip 7! First to 200 points wins!")
        while not self._has_winner():
            self._printf("\n" + "=" * 50)
            self._printf("\n🎯 ROUND %d\n", self._round)
            self._println("=" * 50)
            self._play_round()
            self._show_scores()
            self._next_round()
        winner = self._get_winner()
        self._printf("\n🎉 GAME OVER! %s wins with %d points! 🎉\n", winner.get_name(), winner.get_total_score())

    def _get_int_input(self, min_val: int, max_val: int) -> int:
        while True:
            try:
                line = self._readline().strip()
            except Exception:
                raise RuntimeError("failed to read input")
            try:
                num = int(line)
            except ValueError:
                self._printf("Please enter a valid number between %d and %d: ", min_val, max_val)
                continue
            if num < min_val or num > max_val:
                self._printf("Please enter a number between %d and %d: ", min_val, max_val)
                continue
            return num

    def _get_string_input(self) -> str:
        try:
            return self._readline().strip()
        except Exception:
            raise RuntimeError("failed to read input")

    def _has_winner(self) -> bool:
        return any(p.get_total_score() >= 200 for p in self._players)

    def _get_winner(self) -> BasePlayer:
        return max(self._players, key=lambda p: p.get_total_score())

    def _show_scores(self) -> None:
        self._println("\n📊 Current Scores:")
        self._println("-" * 40)
        for p in self._players:
            self._printf("%s %-20s: %3d points\n", p.get_player_icon(), p.get_name(), p.get_total_score())
        self._println("-" * 40)

    def _next_round(self) -> None:
        self._round += 1
        self._dealer_idx = (self._dealer_idx + 1) % len(self._players)
        for p in self._players:
            for c in p.reset_for_new_round():
                self._deck.discard_card(c)
        total_cards = self._deck.cards_left() + len(self._deck.discards)
        for p in self._players:
            total_cards += len(p.get_hand())
        if total_cards != self._deck.original_total:
            raise RuntimeError(f"Total cards is not the original total. found: {total_cards} != expected: {self._deck.original_total}")

    def _play_round(self) -> None:
        self._printf("Dealer: %s\n\n", self._players[self._dealer_idx].get_name())
        self._deal_initial_cards()
        self._play_turns()
        self._calculate_round_scores()

    def _deal_initial_cards(self) -> None:
        self._println("🃏 Dealing initial cards...")
        n = len(self._players)
        for i in range(n):
            idx = (self._dealer_idx + 1 + i) % n
            player = self._players[idx]
            if not player.is_active():
                continue
            card = self._deck.draw_card()
            self._printf("   %s draws %s\n", player.get_name(), str(card))
            if card.is_action_card():
                self._handle_action_card(player, card)
            else:
                err = player.add_card(card)
                if err:
                    self._handle_card_add_error(player, card, err)
        self._println()
        self._show_all_hands()

    def _play_turns(self) -> None:
        while self._has_active_players():
            n = len(self._players)
            for i in range(n):
                idx = (self._dealer_idx + 1 + i) % n
                player = self._players[idx]
                if not player.is_active():
                    continue
                if not player.has_cards():
                    self._printf("🎯 %s has no number cards and must HIT\n", player.get_name())
                    self._player_hit(player)
                    continue
                should_hit = self._get_player_choice(player)
                if should_hit:
                    self._player_hit(player)
                else:
                    self._player_stay(player)
                if not self._has_active_players():
                    break

    def _calculate_round_scores(self) -> None:
        self._println("📊 Calculating round scores...")
        self._println("-" * 40)
        for p in self._players:
            rs = p.calculate_round_score()
            p.add_to_total_score()
            self._printf("%s: %d points this round (Total: %d)\n", p.get_name(), rs, p.get_total_score())
        self._println("-" * 40)

    def _has_active_players(self) -> bool:
        return any(p.is_active() for p in self._players)

    def _show_all_hands(self) -> None:
        if self._silent_mode:
            return
        for p in self._players:
            p.show_hand()

    def _get_player_choice(self, player: BasePlayer) -> bool:
        return player.make_hit_stay_decision(self._build_game_state())

    def _player_hit(self, player: BasePlayer) -> None:
        card = self._deck.draw_card()
        self._printf("   %s draws %s\n", player.get_name(), str(card))
        if card.is_action_card():
            self._handle_action_card(player, card)
            return
        err = player.add_card(card)
        if err:
            self._handle_card_add_error(player, card, err)

    def _player_stay(self, player: BasePlayer) -> None:
        player.stay()
        self._printf("   %s stays with %d points\n", player.get_name(), player.calculate_round_score())

    def _handle_action_card(self, player: BasePlayer, card: Card) -> None:
        self._printf("   🎲 Action card! %s\n", str(card))
        if card.action == ActionType.FREEZE:
            self._handle_freeze_card(player, card)
        elif card.action == ActionType.FLIP_THREE:
            self._handle_flip_three_card(player, card)
        elif card.action == ActionType.SECOND_CHANCE:
            self._handle_second_chance_card(player, card)

    def _handle_freeze_card(self, player: BasePlayer, card: Card) -> None:
        try:
            target = player.choose_action_target(self._build_game_state(), ActionType.FREEZE)
        except Exception:
            self._deck.discard_card(card)
            raise
        target.stay()
        target.calculate_round_score()
        self._printf("   ❄️ %s is frozen and stays with %d points!\n", target.get_name(), target.calculate_round_score())
        self._deck.discard_card(card)

    def _handle_flip_three_card(self, player: BasePlayer, card: Card) -> None:
        try:
            target = player.choose_action_target(self._build_game_state(), ActionType.FLIP_THREE)
        except Exception:
            self._deck.discard_card(card)
            raise
        self._printf("   🎲 %s must flip 3 cards!\n", target.get_name())
        for i in range(3):
            if not target.is_active():
                break
            drawn = self._deck.draw_card()
            self._printf("      Card %d: %s\n", i + 1, str(drawn))
            if drawn.is_action_card():
                try:
                    self._handle_action_card(target, drawn)
                except Exception as e:
                    if "flip7" in str(e):
                        self._printf("   🎉 %s achieved FLIP 7!\n", target.get_name())
                        self._end_round_for_flip7(target)
                        break
                    self._deck.discard_card(drawn)
                    raise
            else:
                err = target.add_card(drawn)
                if err:
                    self._handle_card_add_error(target, drawn, err)
                    if "flip7" in err or "bust" in err:
                        break
        self._deck.discard_card(card)

    def _handle_second_chance_card(self, player: BasePlayer, card: Card) -> None:
        if not player.has_second_chance():
            self._printf("   🆘 %s receives a Second Chance card!\n", player.get_name())
            err = player.add_card(card)
            if err:
                self._deck.discard_card(card)
                if err != "second_chance_duplicate":
                    raise RuntimeError(err)
            return
        self._printf("   🆘 %s already has Second Chance, must give to another player\n", player.get_name())
        try:
            target = player.choose_positive_action_target(self._build_game_state(), ActionType.SECOND_CHANCE)
        except Exception:
            self._println("   🆘 No one can take the Second Chance card, discarding")
            self._deck.discard_card(card)
            return
        err = target.add_card(card)
        if err:
            self._deck.discard_card(card)
            self._printf("   🆘 %s cannot take the Second Chance card, discarding\n", player.get_name())
            return
        self._printf("   🆘 %s receives a Second Chance card!\n", target.get_name())

    def _handle_card_add_error(self, player: BasePlayer, card: Card, err: str) -> None:
        if "flip7" in err:
            self._printf("   🎉 %s achieved FLIP 7 and wins the round!\n", player.get_name())
            self._end_round_for_flip7(player)
            return
        if "duplicate_with_second_chance" in err:
            self._printf("   💥 %s drew a duplicate %s but has Second Chance!\n", player.get_name(), str(card))
            sc = player.use_second_chance()
            self._deck.discard_card(sc)
            self._deck.discard_card(card)
            return
        if "bust" in err:
            self._deck.discard_card(card)
            self._printf("   💥 %s busts and is out of the round!\n", player.get_name())
            return
        if "second_chance_duplicate" in err:
            new_target = player.choose_positive_action_target(self._build_game_state(), ActionType.SECOND_CHANCE)
            err2 = new_target.add_card(card)
            if err2:
                self._printf("   🆘 %s cannot give the Second Chance card to anyone, discarding\n", player.get_name())
                self._deck.discard_card(card)
            else:
                self._printf("   🆘 %s gives the Second Chance card to %s\n", player.get_name(), new_target.get_name())
            return
        raise RuntimeError(err)

    def _build_game_state(self) -> GameState:
        active = [p for p in self._players if p.is_active()]
        leader: Optional[BasePlayer] = None
        max_score = -1
        for p in self._players:
            s = p.get_total_score() + p.calculate_round_score()
            if s > max_score:
                max_score = s
                leader = p
        return GameState(
            round_num=self._round,
            players=self._players,
            active_players=active,
            current_leader=leader,
            cards_in_deck=self._deck.cards,
        )

    def _end_round_for_flip7(self, flip7_player: BasePlayer) -> None:
        for p in self._players:
            if p != flip7_player and p.is_active():
                p.stay()
                p.calculate_round_score()

    def _setup_players(self) -> None:
        if self._agent_n_players is not None:
            self._setup_players_agent_mode()
            return
        self._println("How many players total? (2-18): ")
        num_players = self._get_int_input(2, 18)
        self._printf("How many human players? (0-%d): ", num_players)
        num_humans = self._get_int_input(0, num_players)
        num_computers = num_players - num_humans
        for i in range(num_humans):
            self._printf("Enter name for Human Player %d: ", i + 1)
            name = self._get_string_input()
            self._players.append(HumanPlayer(name, self._readline))
        names_pool = list(COMPUTER_NAMES)
        for i in range(num_computers):
            name, hit_stay, action_tgt, positive_tgt = self._get_computer_player_setup(i + 1, names_pool)
            self._players.append(ComputerPlayer(name, hit_stay, action_tgt, positive_tgt))
            self._printf("  → Added: %s (%s AI)\n", name, self._players[-1].get_name())
        if num_humans == 0:
            self._printf("\n🎮 Starting AI-only Flip 7 with %d computer players!\n", num_computers)
            self._println("🍿 Sit back and watch the AIs battle it out!")
            self._printf("\nHow many games would you like to simulate? ")
            num_games = self._get_int_input(1, 2 ** 31 - 1)
            if num_games > 1:
                self.set_silent_mode(True)
                self._run_multiple_games(num_games)
                return
        else:
            self._printf("\n🎮 Starting Flip 7 with %d humans and %d computers!\n", num_humans, num_computers)

    def _setup_players_agent_mode(self) -> None:
        from rl_agent_player import RLPlayer
        from rl_env import Flip7Env
        num_players = self._agent_n_players
        self._printf("How many human players? (0-%d): ", num_players)
        num_humans = self._get_int_input(0, num_players)
        if num_humans == 1:
            self._players.append(HumanPlayer("Human", self._readline))
        else:
            for i in range(num_humans):
                self._printf("Enter name for Human Player %d: ", i + 1)
                name = self._get_string_input()
                self._players.append(HumanPlayer(name, self._readline))
        self._agent_env = Flip7Env()
        num_agents = num_players - num_humans
        for i in range(num_agents):
            player_idx = num_humans + i
            pl = RLPlayer(
                f"Agent{i}",
                self._agent_network,
                player_idx,
                self._agent_env,
                is_training_agent=False,
                show_obs_and_head=True,
            )
            self._players.append(pl)
            self._printf("  → Added: %s (checkpoint agent)\n", pl.get_name())
        self._printf("\n🎮 Starting Flip 7 with %d human(s) and %d checkpoint agent(s)!\n", num_humans, num_agents)

    def _get_computer_player_setup(
        self, computer_num: int, names_pool: List[str]
    ) -> tuple:
        if not names_pool:
            name = f"AI{computer_num}"
        else:
            name = names_pool.pop(random.randint(0, len(names_pool) - 1))
        self._printf("\nComputer Player %d:\n", computer_num)
        self._println("Choose AI strategy:")
        self._println("  1) Plays to 20")
        self._println("  2) Plays to 25")
        self._println("  3) Plays to 30")
        self._println("  4) Plays to 35")
        self._println("  5) Hit p(BUST) < 0.2")
        self._println("  6) Hit p(BUST) < 0.25")
        self._println("  7) Hit p(BUST) < 0.3")
        self._println("  8) Hit p(BUST) < 0.35")
        self._println("  9) Hit p(BUST) < 0.4")
        self._println("  10) FLIP 7")
        self._println("  11) Random")
        self._println("  12) Adaptive Bust Prob (0.3)")
        self._println("  13) Expected Value")
        self._println("  14) Hybrid Strategy")
        self._println("  15) Gap-Based Strategy")
        self._println("  16) Optimal Strategy")
        self._print("Enter choice (1-16): ")
        try:
            choice = self._get_int_input(1, 16)
        except Exception:
            choice = 13
        action_tgt = target_leader_strategy
        positive_tgt = target_last_place_strategy
        if choice == 1:
            name += " (20)"
            hit_stay = play_round_to(20)
        elif choice == 2:
            name += " (25)"
            hit_stay = play_round_to(25)
        elif choice == 3:
            name += " (30)"
            hit_stay = play_round_to(30)
        elif choice == 4:
            name += " (35)"
            hit_stay = play_round_to(35)
        elif choice == 5:
            name += " (p0.2)"
            hit_stay = play_to_bust_probability(0.2)
        elif choice == 6:
            name += " (p0.25)"
            hit_stay = play_to_bust_probability(0.25)
        elif choice == 7:
            name += " (p0.3)"
            hit_stay = play_to_bust_probability(0.3)
        elif choice == 8:
            name += " (p0.35)"
            hit_stay = play_to_bust_probability(0.35)
        elif choice == 9:
            name += " (p0.4)"
            hit_stay = play_to_bust_probability(0.4)
        elif choice == 10:
            name += " (hit)"
            hit_stay = always_hit_strategy
        elif choice == 11:
            name += " (rand)"
            hit_stay = random_hit_or_stay_strategy
            action_tgt = target_random_strategy
            positive_tgt = target_random_strategy
        elif choice == 12:
            name += " (adapt0.3)"
            hit_stay = adaptive_bust_probability_strategy(0.3)
        elif choice == 13:
            name += " (exp)"
            hit_stay = expected_value_strategy
        elif choice == 14:
            name += " (hybrid)"
            hit_stay = hybrid_strategy
        elif choice == 15:
            name += " (gap)"
            hit_stay = gap_based_strategy
        elif choice == 16:
            name += " (opt)"
            hit_stay = optimal_strategy
        else:
            hit_stay = expected_value_strategy
        return name, hit_stay, action_tgt, positive_tgt

    def _run_multiple_games(self, num_games: int) -> None:
        self._printf("\n🎲 Running %d games for statistical analysis...\n", num_games)
        player_wins: dict = {p.get_name(): 0 for p in self._players}
        player_names = [p.get_name() for p in self._players]
        import time
        start = time.time()
        last_progress = start
        for game_num in range(1, num_games + 1):
            now = time.time()
            if game_num == 1 or now - last_progress >= 5:
                self._printf("⚡ Game %d/%d... (%.1fs elapsed)\n", game_num, num_games, now - start)
                last_progress = now
            self._reset_game_state()
            self.set_silent_mode(True)
            self._run_single_game()
            winner = self._get_winner()
            player_wins[winner.get_name()] = player_wins.get(winner.get_name(), 0) + 1
            self.set_silent_mode(False)
        self._display_game_statistics(num_games, player_wins, player_names)

    def _reset_game_state(self) -> None:
        self._round = 1
        self._dealer_idx = 0
        for p in self._players:
            for c in p.reset_for_new_round():
                self._deck.discard_card(c)
            p.total_score = 0
        self._deck = Deck()

    def _run_single_game(self) -> None:
        while not self._has_winner():
            self._play_round()
            self._next_round()

    def _display_game_statistics(
        self, num_games: int, player_wins: dict, player_names: list
    ) -> None:
        self._println("\n" + "=" * 60)
        self._printf("🏆 SIMULATION RESULTS - %d GAMES COMPLETED\n", num_games)
        self._println("=" * 60)
        stats = [(name, player_wins.get(name, 0), player_wins.get(name, 0) / num_games * 100) for name in player_names]
        stats.sort(key=lambda x: -x[1])
        self._printf("%-20s %8s %10s %12s\n", "PLAYER", "WINS", "WIN RATE", "PERFORMANCE")
        self._println("-" * 60)
        for i, (name, wins, rate) in enumerate(stats):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            if rate >= 50:
                perf = "🔥 DOMINANT"
            elif rate >= 35:
                perf = "💪 STRONG"
            elif rate >= 20:
                perf = "👍 DECENT"
            else:
                perf = "😔 WEAK"
            self._printf("%-20s %8d %9.1f%% %12s %s\n", name, wins, rate, perf, medal)
        self._println("-" * 60)
        self._printf("Total Games: %d\n", num_games)
        if len(stats) > 1:
            margin = stats[0][2] - stats[1][2]
            self._printf("Victory Margin: %.1f%% (%s vs %s)\n", margin, stats[0][0], stats[1][0])
        self._println("=" * 60)
