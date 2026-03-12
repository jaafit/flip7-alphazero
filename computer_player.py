"""Computer player and AI strategies for Flip 7."""

import math
import random
from typing import Callable, List, Optional

from card import Card, CardType, ActionType, ModifierType
from player import BasePlayer, GameState


HitOrStayStrategy = Callable[[BasePlayer, GameState], bool]
ActionTargetStrategy = Callable[[BasePlayer, GameState, ActionType], BasePlayer]


def calculate_bust_probability(player: BasePlayer, game_state: GameState) -> float:
    """Probability that the next card causes a bust."""
    number_cards = {c.value for c in player.get_hand() if c.type == CardType.NUMBER}
    bust_cards = sum(
        1 for c in game_state.cards_in_deck
        if c.type == CardType.NUMBER and c.value in number_cards
    )
    total = len(game_state.cards_in_deck)
    if total == 0:
        raise RuntimeError("no cards left in deck can't calculate bust probability")
    return bust_cards / total


def calculate_expected_points_from_hit(player: BasePlayer, game_state: GameState) -> float:
    """Expected points from drawing one more card."""
    number_cards = {c.value for c in player.get_hand() if c.type == CardType.NUMBER}
    total_points = 0.0
    valid_cards = 0
    for c in game_state.cards_in_deck:
        if c.type == CardType.NUMBER and c.value not in number_cards:
            total_points += c.value
            valid_cards += 1
        elif c.type == CardType.MODIFIER:
            total_points += c.get_points()
            valid_cards += 1
        elif c.type == CardType.ACTION:
            total_points += 5.0
            valid_cards += 1
    if valid_cards == 0:
        return 0.0
    return total_points / valid_cards


def _has_multiplier(player: BasePlayer) -> bool:
    for c in player.get_hand():
        if c.type == CardType.MODIFIER and c.modifier == ModifierType.MULTIPLY_2:
            return True
    return False


# --- Hit/Stay strategies ---

def play_round_to(n: int) -> HitOrStayStrategy:
    def strategy(self: BasePlayer, game_state: GameState) -> bool:
        return self.calculate_round_score() < n
    return strategy


def play_to_bust_probability(p: float) -> HitOrStayStrategy:
    def strategy(self: BasePlayer, game_state: GameState) -> bool:
        return calculate_bust_probability(self, game_state) < p
    return strategy


def hit_until_ahead_by(n: int) -> HitOrStayStrategy:
    def strategy(self: BasePlayer, game_state: GameState) -> bool:
        leader = game_state.current_leader
        if leader is None:
            return True
        leader_score = leader.get_total_score() + leader.calculate_round_score()
        my_score = self.get_total_score() + self.calculate_round_score()
        return leader_score < my_score + n
    return strategy


def always_hit_strategy(self: BasePlayer, game_state: GameState) -> bool:
    return True


def random_hit_or_stay_strategy(self: BasePlayer, game_state: GameState) -> bool:
    return random.randint(0, 1) == 0


def adaptive_bust_probability_strategy(base_probability: float) -> HitOrStayStrategy:
    def strategy(self: BasePlayer, game_state: GameState) -> bool:
        bust_prob = calculate_bust_probability(self, game_state)
        adjusted = base_probability
        leader = game_state.current_leader
        if leader is not None and leader != self:
            leader_score = leader.get_total_score() + leader.calculate_round_score()
            my_score = self.get_total_score() + self.calculate_round_score()
            gap = leader_score - my_score
            if gap > 50:
                adjusted += 0.15
            elif gap > 20:
                adjusted += 0.1
            elif gap < -20:
                adjusted -= 0.1
        if leader is not None and leader.get_total_score() > 150:
            adjusted += 0.1
        return bust_prob < adjusted
    return strategy


def expected_value_strategy(self: BasePlayer, game_state: GameState) -> bool:
    bust_prob = calculate_bust_probability(self, game_state)
    expected_points = calculate_expected_points_from_hit(self, game_state)
    current_score = self.calculate_round_score()
    expected_value = expected_points * (1 - bust_prob)
    threshold = 2.0
    leader = game_state.current_leader
    if leader is not None and leader != self:
        leader_score = leader.get_total_score() + leader.calculate_round_score()
        my_score = self.get_total_score() + current_score
        if leader_score - my_score > 30:
            threshold = 1.0
    return expected_value > threshold and bust_prob < 0.5


def hybrid_strategy(self: BasePlayer, game_state: GameState) -> bool:
    current_score = self.calculate_round_score()
    bust_prob = calculate_bust_probability(self, game_state)
    base_bust_threshold = 0.25
    leader = game_state.current_leader
    if leader is not None:
        leader_score = leader.get_total_score() + leader.calculate_round_score()
        my_score = self.get_total_score() + current_score
        gap = leader_score - my_score
        if gap > 40:
            base_bust_threshold += 0.2
        elif gap > 15:
            base_bust_threshold += 0.1
        elif gap < -15:
            base_bust_threshold -= 0.1
    if current_score > 30:
        base_bust_threshold -= 0.1
    elif current_score < 10:
        base_bust_threshold += 0.05
    if leader is not None and leader.get_total_score() > 160:
        base_bust_threshold += 0.1
    if _has_multiplier(self):
        base_bust_threshold += 0.05
    return bust_prob < base_bust_threshold


def gap_based_strategy(self: BasePlayer, game_state: GameState) -> bool:
    leader = game_state.current_leader
    if leader is None:
        return calculate_bust_probability(self, game_state) < 0.3
    leader_score = leader.get_total_score() + leader.calculate_round_score()
    my_score = self.get_total_score() + self.calculate_round_score()
    gap = leader_score - my_score
    if gap > 60:
        bust_threshold = 0.5
    elif gap > 30:
        bust_threshold = 0.4
    elif gap > 10:
        bust_threshold = 0.35
    elif gap > -10:
        bust_threshold = 0.3
    elif gap > -30:
        bust_threshold = 0.25
    else:
        bust_threshold = 0.2
    return calculate_bust_probability(self, game_state) < bust_threshold


def optimal_strategy(self: BasePlayer, game_state: GameState) -> bool:
    bust_prob = calculate_bust_probability(self, game_state)
    current_score = self.calculate_round_score()
    leader = game_state.current_leader
    if leader is not None:
        leader_score = leader.get_total_score() + leader.calculate_round_score()
        my_score = self.get_total_score() + current_score
        gap = leader_score - my_score
        if gap > 50:
            base_threshold = 0.45
        elif gap > 25:
            base_threshold = 0.35
        elif gap > 10:
            base_threshold = 0.3
        elif gap > -10:
            base_threshold = 0.28
        elif gap > -25:
            base_threshold = 0.25
        else:
            base_threshold = 0.22
    else:
        base_threshold = 0.3
    if current_score > 35:
        base_threshold -= 0.08
    elif current_score > 25:
        base_threshold -= 0.05
    elif current_score < 10:
        base_threshold += 0.03
    if leader is not None:
        max_score = leader.get_total_score()
        if max_score > 170:
            base_threshold += 0.05
        elif max_score > 150:
            base_threshold += 0.03
    if _has_multiplier(self) and current_score < 25:
        base_threshold += 0.04
    base_threshold = max(0.15, min(0.5, base_threshold))
    return bust_prob < base_threshold


# --- Action target strategies ---

def target_leader_strategy(
    self: BasePlayer, game_state: GameState, action_type: ActionType
) -> BasePlayer:
    leader: Optional[BasePlayer] = None
    leader_score = 0
    for p in game_state.active_players:
        if action_type == ActionType.SECOND_CHANCE and p.has_second_chance():
            continue
        if p != self:
            s = p.get_total_score() + p.calculate_round_score()
            if s > leader_score:
                leader = p
                leader_score = s
    return leader if leader is not None else self


def target_last_place_strategy(
    self: BasePlayer, game_state: GameState, action_type: ActionType
) -> BasePlayer:
    last: Optional[BasePlayer] = None
    last_score: int = 2 ** 31 - 1  # maxint-like
    for p in game_state.active_players:
        if action_type == ActionType.SECOND_CHANCE and p.has_second_chance():
            continue
        if p != self:
            s = p.get_total_score() + p.calculate_round_score()
            if s < last_score:
                last = p
                last_score = s
    return last if last is not None else self


def target_random_strategy(
    self: BasePlayer, game_state: GameState, action_type: ActionType
) -> BasePlayer:
    active: List[BasePlayer] = []
    for p in game_state.players:
        if action_type == ActionType.SECOND_CHANCE and p.has_second_chance():
            continue
        if p.is_active() and p != self:
            active.append(p)
    if not active:
        return self
    return random.choice(active)


class ComputerPlayer(BasePlayer):
    """Computer player that uses configurable hit/stay and action target strategies."""

    def __init__(
        self,
        name: str,
        hit_or_stay_strategy: HitOrStayStrategy,
        action_target_strategy: ActionTargetStrategy,
        positive_action_target_strategy: ActionTargetStrategy,
    ) -> None:
        super().__init__(name)
        self.init(name)
        self._hit_or_stay = hit_or_stay_strategy
        self._action_target = action_target_strategy
        self._positive_action_target = positive_action_target_strategy

    def get_player_icon(self) -> str:
        return "🤖"

    def make_hit_stay_decision(self, game_state: GameState) -> bool:
        if self.has_second_chance():
            return True
        return self._hit_or_stay(self, game_state)

    def choose_action_target(
        self, game_state: GameState, action_type: ActionType
    ) -> BasePlayer:
        return self._action_target(self, game_state, action_type)

    def choose_positive_action_target(
        self, game_state: GameState, action_type: ActionType
    ) -> BasePlayer:
        return self._positive_action_target(self, game_state, action_type)
