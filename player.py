"""Player state and hand management for Flip 7."""

from __future__ import annotations

from enum import IntEnum
from typing import List, Optional

from card import Card, CardType, ActionType, ModifierType


class PlayerState(IntEnum):
    ACTIVE = 0
    STAYED = 1
    BUSTED = 2


class PlayerType(IntEnum):
    HUMAN = 0
    COMPUTER = 1


class GameState:
    """Context for AI decision making."""

    def __init__(
        self,
        round_num: int,
        players: List[BasePlayer],
        active_players: List[BasePlayer],
        current_leader: Optional[BasePlayer],
        cards_in_deck: List[Card],
    ) -> None:
        self.round = round_num
        self.players = players
        self.active_players = active_players
        self.current_leader = current_leader
        self.cards_in_deck = cards_in_deck


class BasePlayer:
    """Base player state and hand logic."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.total_score = 0
        self.number_cards: List[Card] = []
        self.modifier_cards: List[Card] = []
        self.action_cards: List[Card] = []
        self.state = PlayerState.ACTIVE
        self.second_chance = False

    def init(self, name: str) -> None:
        self.name = name
        self.number_cards = []
        self.modifier_cards = []
        self.action_cards = []
        self.state = PlayerState.ACTIVE

    def get_name(self) -> str:
        return self.name

    def has_second_chance(self) -> bool:
        return self.second_chance

    def get_total_score(self) -> int:
        return self.total_score

    def add_card(self, card: Card) -> Optional[str]:
        """Add a card to the player's hand. Returns error string or None."""
        if card.type == CardType.NUMBER:
            for existing in self.number_cards:
                if existing.value == card.value:
                    if self.has_second_chance():
                        return f"duplicate_with_second_chance:{card.value}"
                    self.bust()
                    return f"bust:{card.value}"
            self.number_cards.append(card)
            if len(self.number_cards) == 7:
                self.stay()
                return "flip7"
            return None

        if card.type == CardType.MODIFIER:
            self.modifier_cards.append(card)
            return None
        if card.type == CardType.ACTION:
            if card.action == ActionType.SECOND_CHANCE:
                if self.has_second_chance():
                    return "second_chance_duplicate"
                self.second_chance = True
            self.action_cards.append(card)
            return None
        return None

    def get_hand(self) -> List[Card]:
        return self.number_cards + self.modifier_cards + self.action_cards

    def use_second_chance(self) -> Card:
        if not self.has_second_chance():
            raise RuntimeError("no second chance card to use")
        self.second_chance = False
        for i, card in enumerate(self.action_cards):
            if card.action == ActionType.SECOND_CHANCE:
                self.action_cards.pop(i)
                return card
        raise RuntimeError("no second chance card to use")

    def stay(self) -> None:
        if self.state == PlayerState.ACTIVE:
            self.state = PlayerState.STAYED
        else:
            raise RuntimeError("stay when player is not active")

    def bust(self) -> None:
        if self.state == PlayerState.ACTIVE:
            self.state = PlayerState.BUSTED
        else:
            raise RuntimeError("bust when player is not active")

    def calculate_round_score(self) -> int:
        if self.state == PlayerState.BUSTED:
            return 0
        number_total = sum(c.value for c in self.number_cards)
        for c in self.modifier_cards:
            if c.modifier == ModifierType.MULTIPLY_2:
                number_total *= 2
                break
        modifier_total = sum(
            c.get_points() for c in self.modifier_cards
            if c.modifier != ModifierType.MULTIPLY_2
        )
        total = number_total + modifier_total
        if len(self.number_cards) == 7:
            total += 15
        return total

    def add_to_total_score(self) -> None:
        self.total_score += self.calculate_round_score()

    def reset_for_new_round(self) -> List[Card]:
        discarded = self.get_hand()
        self.number_cards = []
        self.modifier_cards = []
        self.action_cards = []
        self.state = PlayerState.ACTIVE
        self.second_chance = False
        return discarded

    def is_active(self) -> bool:
        return self.state == PlayerState.ACTIVE

    def has_cards(self) -> bool:
        return len(self.number_cards) > 0

    def show_hand(self) -> None:
        print(f"{self.name}:")
        if not self.number_cards and not self.modifier_cards:
            print("   No cards")
            return
        if self.number_cards:
            print("   Numbers: ", " ".join(str(c) for c in self.number_cards))
        if self.modifier_cards:
            print("   Modifiers: ", " ".join(str(c) for c in self.modifier_cards))
        if self.has_second_chance():
            print("   🆘 Has Second Chance")
        if self.state == PlayerState.STAYED:
            print(f"   ✅ STAYED - Round Score: {self.calculate_round_score()}")
        if self.state == PlayerState.BUSTED:
            print("   💥 BUSTED")
        print()

    def get_hand_summary(self) -> str:
        if self.state == PlayerState.BUSTED:
            return "💥 BUSTED"
        if not self.number_cards and not self.modifier_cards:
            return "No cards"
        parts = []
        if self.number_cards:
            parts.append(",".join(str(c.value) for c in self.number_cards))
        if self.modifier_cards:
            parts.append(",".join(str(c) for c in self.modifier_cards))
        result = " | ".join(parts)
        if self.state == PlayerState.STAYED:
            result += f" (STAYED: {self.calculate_round_score()} pts)"
        return result
