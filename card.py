"""Card types and definitions for Flip 7."""

from enum import IntEnum
from dataclasses import dataclass
from typing import Optional


class CardType(IntEnum):
    """Types of cards in Flip 7."""
    NUMBER = 0
    ACTION = 1
    MODIFIER = 2


class ActionType(IntEnum):
    """Action card types."""
    FREEZE = 0
    FLIP_THREE = 1
    SECOND_CHANCE = 2


class ModifierType(IntEnum):
    """Modifier card types."""
    PLUS_2 = 0
    PLUS_4 = 1
    PLUS_6 = 2
    PLUS_8 = 3
    PLUS_10 = 4
    MULTIPLY_2 = 5


@dataclass
class Card:
    """A single card in the game."""
    type: CardType
    value: int = 0  # For number cards (0-12)
    action: Optional[ActionType] = None  # For action cards
    modifier: Optional[ModifierType] = None  # For modifier cards

    @staticmethod
    def new_number_card(value: int) -> "Card":
        return Card(type=CardType.NUMBER, value=value)

    @staticmethod
    def new_action_card(action: ActionType) -> "Card":
        return Card(type=CardType.ACTION, action=action)

    @staticmethod
    def new_modifier_card(modifier: ModifierType) -> "Card":
        return Card(type=CardType.MODIFIER, modifier=modifier)

    def __str__(self) -> str:
        if self.type == CardType.NUMBER:
            return f"[{self.value}]"
        if self.type == CardType.ACTION:
            if self.action == ActionType.FREEZE:
                return "[❄️ FREEZE]"
            if self.action == ActionType.FLIP_THREE:
                return "[🎲 FLIP 3]"
            if self.action == ActionType.SECOND_CHANCE:
                return "[🆘 2ND CHANCE]"
        if self.type == CardType.MODIFIER:
            if self.modifier == ModifierType.PLUS_2:
                return "[+2]"
            if self.modifier == ModifierType.PLUS_4:
                return "[+4]"
            if self.modifier == ModifierType.PLUS_6:
                return "[+6]"
            if self.modifier == ModifierType.PLUS_8:
                return "[+8]"
            if self.modifier == ModifierType.PLUS_10:
                return "[+10]"
            if self.modifier == ModifierType.MULTIPLY_2:
                return "[×2]"
        return "[?]"

    def get_points(self) -> int:
        """Return the point value of the card."""
        if self.type == CardType.NUMBER:
            return self.value
        if self.type == CardType.MODIFIER:
            if self.modifier == ModifierType.PLUS_2:
                return 2
            if self.modifier == ModifierType.PLUS_4:
                return 4
            if self.modifier == ModifierType.PLUS_6:
                return 6
            if self.modifier == ModifierType.PLUS_8:
                return 8
            if self.modifier == ModifierType.PLUS_10:
                return 10
            if self.modifier == ModifierType.MULTIPLY_2:
                return 0  # Multiplier doesn't add points directly
        return 0

    def is_number_card(self) -> bool:
        return self.type == CardType.NUMBER

    def is_action_card(self) -> bool:
        return self.type == CardType.ACTION

    def is_modifier_card(self) -> bool:
        return self.type == CardType.MODIFIER

    def can_cause_bust(self) -> bool:
        return self.type == CardType.NUMBER
