package main

import "fmt"

// CardType represents the different types of cards in Flip 7
type CardType int

const (
	NumberCard CardType = iota
	ActionCard
	ModifierCard
)

// ActionType represents the different action cards
type ActionType int

const (
	Freeze ActionType = iota
	FlipThree
	SecondChance
)

// ModifierType represents the different modifier cards
type ModifierType int

const (
	Plus2 ModifierType = iota
	Plus4
	Plus6
	Plus8
	Plus10
	Multiply2
)

// Card represents a single card in the game
type Card struct {
	Type     CardType
	Value    int          // For number cards (0-12)
	Action   ActionType   // For action cards
	Modifier ModifierType // For modifier cards
}

// NewNumberCard creates a new number card
func NewNumberCard(value int) *Card {
	return &Card{
		Type:  NumberCard,
		Value: value,
	}
}

// NewActionCard creates a new action card
func NewActionCard(action ActionType) *Card {
	return &Card{
		Type:   ActionCard,
		Action: action,
	}
}

// NewModifierCard creates a new modifier card
func NewModifierCard(modifier ModifierType) *Card {
	return &Card{
		Type:     ModifierCard,
		Modifier: modifier,
	}
}

// String returns a string representation of the card
func (c *Card) String() string {
	switch c.Type {
	case NumberCard:
		return fmt.Sprintf("[%d]", c.Value)
	case ActionCard:
		switch c.Action {
		case Freeze:
			return "[❄️ FREEZE]"
		case FlipThree:
			return "[🎲 FLIP 3]"
		case SecondChance:
			return "[🆘 2ND CHANCE]"
		}
	case ModifierCard:
		switch c.Modifier {
		case Plus2:
			return "[+2]"
		case Plus4:
			return "[+4]"
		case Plus6:
			return "[+6]"
		case Plus8:
			return "[+8]"
		case Plus10:
			return "[+10]"
		case Multiply2:
			return "[×2]"
		}
	}
	return "[?]"
}

// GetPoints returns the point value of the card
func (c *Card) GetPoints() int {
	switch c.Type {
	case NumberCard:
		return c.Value
	case ModifierCard:
		switch c.Modifier {
		case Plus2:
			return 2
		case Plus4:
			return 4
		case Plus6:
			return 6
		case Plus8:
			return 8
		case Plus10:
			return 10
		case Multiply2:
			return 0 // Multiplier doesn't add points directly
		}
	}
	return 0
}

// IsNumberCard checks if the card is a number card
func (c *Card) IsNumberCard() bool {
	return c.Type == NumberCard
}

// IsActionCard checks if the card is an action card
func (c *Card) IsActionCard() bool {
	return c.Type == ActionCard
}

// IsModifierCard checks if the card is a modifier card
func (c *Card) IsModifierCard() bool {
	return c.Type == ModifierCard
}

// CanCauseBust checks if this card can cause a player to bust
func (c *Card) CanCauseBust() bool {
	return c.Type == NumberCard
}
