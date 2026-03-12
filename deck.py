"""Deck management and shuffling for Flip 7."""

import random
from typing import Callable, List, Optional

from card import (
    Card,
    CardType,
    ActionType,
    ModifierType,
)


class Deck:
    """The game deck."""

    def __init__(self) -> None:
        self._cards: List[Card] = []
        self._discards: List[Card] = []
        self._rng = random.Random()
        self._debug_mode = False
        self._scanner = None  # Set via set_debug_mode; type: Optional[any]
        self.original_total = 0
        self._create_cards()
        self.shuffle()
        self.original_total = len(self._cards)

    def _create_cards(self) -> None:
        """Build all cards with correct distribution."""
        # Number cards: each number has as many cards as its value
        for value in range(13):  # 0..12
            count = value if value > 0 else 1
            for _ in range(count):
                self._cards.append(Card.new_number_card(value))

        # Score modifier cards (6 total)
        for mod in (
            ModifierType.PLUS_2,
            ModifierType.PLUS_4,
            ModifierType.PLUS_6,
            ModifierType.PLUS_8,
            ModifierType.PLUS_10,
            ModifierType.MULTIPLY_2,
        ):
            self._cards.append(Card.new_modifier_card(mod))

        # Action cards (3 of each type = 9 total)
        for _ in range(3):
            for action in (ActionType.FREEZE, ActionType.FLIP_THREE, ActionType.SECOND_CHANCE):
                self._cards.append(Card.new_action_card(action))

    def shuffle(self) -> None:
        self._rng.shuffle(self._cards)

    def draw_card(self) -> Card:
        if not self._cards:
            raise RuntimeError("All cards disappeared!")
        if self._debug_mode:
            return self._draw_card_debug()

        card = self._cards.pop()
        if not self._cards:
            self._reshuffle()
        return card

    def discard_card(self, card: Optional[Card]) -> None:
        if card is not None:
            self._discards.append(card)

    def _reshuffle(self) -> None:
        self._cards.extend(self._discards)
        self._discards = []
        self.shuffle()

    def cards_left(self) -> int:
        return len(self._cards)

    def total_cards(self) -> int:
        return len(self._cards) + len(self._discards)

    def set_debug_mode(self, debug: bool, scanner: Optional[Callable[[], str]] = None) -> None:
        self._debug_mode = debug
        self._scanner = scanner

    def _draw_card_debug(self) -> Card:
        if not self._cards:
            raise RuntimeError("No cards in deck")
        number_cards: List[Card] = []
        action_cards: List[Card] = []
        modifier_cards: List[Card] = []
        for c in self._cards:
            if c.type == CardType.NUMBER:
                number_cards.append(c)
            elif c.type == CardType.ACTION:
                action_cards.append(c)
            else:
                modifier_cards.append(c)

        card_options: List[Card] = []
        option_index = 1

        print("\n🐛 DEBUG: Choose a card to draw:")
        print(f"Available cards ({len(self._cards)} total):")

        if number_cards:
            print("\nNumber Cards:")
            card_counts: dict[int, int] = {}
            for c in number_cards:
                card_counts[c.value] = card_counts.get(c.value, 0) + 1
            for value in range(13):
                count = card_counts.get(value, 0)
                if count > 0:
                    print(f"  {option_index}) [{value}] ({count} available)")
                    card_options.append(Card.new_number_card(value))
                    option_index += 1

        if action_cards:
            print("\nAction Cards:")
            action_names = ["❄️ FREEZE", "🎲 FLIP 3", "🆘 2ND CHANCE"]
            for i, at in enumerate([ActionType.FREEZE, ActionType.FLIP_THREE, ActionType.SECOND_CHANCE]):
                count = sum(1 for c in action_cards if c.action == at)
                if count > 0:
                    print(f"  {option_index}) {action_names[i]} ({count} available)")
                    card_options.append(Card.new_action_card(at))
                    option_index += 1

        if modifier_cards:
            print("\nModifier Cards:")
            mod_names = ["+2", "+4", "+6", "+8", "+10", "×2"]
            for i, mt in enumerate(ModifierType):
                count = sum(1 for c in modifier_cards if c.modifier == mt)
                if count > 0:
                    print(f"  {option_index}) [{mod_names[i]}] ({count} available)")
                    card_options.append(Card.new_modifier_card(mt))
                    option_index += 1

        n = len(card_options)
        print(f"\nEnter choice (1-{n}): ", end="", flush=True)

        while True:
            if self._scanner is None:
                return self._draw_random_card()
            try:
                line = self._scanner()
            except Exception:
                return self._draw_random_card()
            line = (line or "").strip()
            try:
                choice = int(line)
            except ValueError:
                print(f"Please enter a number between 1 and {n}: ", end="", flush=True)
                continue
            if choice < 1 or choice > n:
                print(f"Please enter a number between 1 and {n}: ", end="", flush=True)
                continue

            selected = card_options[choice - 1]
            for i, c in enumerate(self._cards):
                if self._cards_equal(c, selected):
                    return self._cards.pop(i)
            print("Card not found, drawing random card instead...")
            return self._draw_random_card()

    def _draw_random_card(self) -> Card:
        if not self._cards:
            raise RuntimeError("No cards in deck")
        return self._cards.pop()

    def _cards_equal(self, c1: Card, c2: Card) -> bool:
        if c1.type != c2.type:
            return False
        if c1.type == CardType.NUMBER:
            return c1.value == c2.value
        if c1.type == CardType.ACTION:
            return c1.action == c2.action
        if c1.type == CardType.MODIFIER:
            return c1.modifier == c2.modifier
        return False

    # Expose for game's card count checks
    @property
    def cards(self) -> List[Card]:
        return self._cards

    @property
    def discards(self) -> List[Card]:
        return self._discards
