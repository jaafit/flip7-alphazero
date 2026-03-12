package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"strconv"
	"strings"
	"time"
)

// Deck represents the game deck
type Deck struct {
	cards         []*Card
	discards      []*Card
	rng           *rand.Rand
	debugMode     bool
	scanner       *bufio.Scanner
	OriginalTotal int
}

// NewDeck creates a new deck with the correct card distribution for Flip 7
func NewDeck() *Deck {
	deck := &Deck{
		cards:    make([]*Card, 0),
		discards: make([]*Card, 0),
		rng:      rand.New(rand.NewSource(time.Now().UnixNano())),
	}

	deck.createCards()
	deck.Shuffle()
	deck.OriginalTotal = len(deck.cards)

	return deck
}

// createCards creates all cards with the correct distributions
func (d *Deck) createCards() {
	// Number cards: each number has as many cards as its value
	// 12 has 12 copies, 11 has 11 copies, etc., down to 0 which has 1 copy
	for value := 0; value <= 12; value++ {
		count := value
		if value == 0 {
			count = 1
		}
		for i := 0; i < count; i++ {
			d.cards = append(d.cards, NewNumberCard(value))
		}
	}

	// Score Modifier Cards (6 total)
	d.cards = append(d.cards, NewModifierCard(Plus2))
	d.cards = append(d.cards, NewModifierCard(Plus4))
	d.cards = append(d.cards, NewModifierCard(Plus6))
	d.cards = append(d.cards, NewModifierCard(Plus8))
	d.cards = append(d.cards, NewModifierCard(Plus10))
	d.cards = append(d.cards, NewModifierCard(Multiply2))

	// Action Cards (3 of each type = 9 total)
	for i := 0; i < 3; i++ {
		d.cards = append(d.cards, NewActionCard(Freeze))
		d.cards = append(d.cards, NewActionCard(FlipThree))
		d.cards = append(d.cards, NewActionCard(SecondChance))
	}
}

// Shuffle shuffles the deck
func (d *Deck) Shuffle() {
	d.rng.Shuffle(len(d.cards), func(i, j int) {
		d.cards[i], d.cards[j] = d.cards[j], d.cards[i]
	})
}

// DrawCard draws the top card from the deck
func (d *Deck) DrawCard() *Card {
	if len(d.cards) == 0 {
		panic("All cards disappeared!	")
	}
	if d.debugMode {
		return d.drawCardDebug()
	}

	card := d.cards[len(d.cards)-1]
	d.cards = d.cards[:len(d.cards)-1]

	if len(d.cards) == 0 {
		d.Reshuffle()
	}

	return card
}

// DiscardCard adds a card to the discard pile
func (d *Deck) DiscardCard(card *Card) {
	if card != nil {
		d.discards = append(d.discards, card)
	}
}

// Reshuffle reshuffles the discard pile back into the deck
func (d *Deck) Reshuffle() {
	d.cards = append(d.cards, d.discards...)
	d.discards = make([]*Card, 0)
	d.Shuffle()
}

// CardsLeft returns the number of cards remaining in the deck
func (d *Deck) CardsLeft() int {
	return len(d.cards)
}

// TotalCards returns the total number of cards (deck + discards)
func (d *Deck) TotalCards() int {
	return len(d.cards) + len(d.discards)
}

// SetDebugMode enables or disables debug mode for manual card selection
func (d *Deck) SetDebugMode(debug bool, scanner *bufio.Scanner) {
	d.debugMode = debug
	d.scanner = scanner
}

// drawCardDebug allows manual selection of cards in debug mode
func (d *Deck) drawCardDebug() *Card {
	if len(d.cards) == 0 {
		return nil
	}

	fmt.Println("\n🐛 DEBUG: Choose a card to draw:")
	fmt.Printf("Available cards (%d total):\n", len(d.cards))

	// Group cards by type for easier selection
	numberCards := make([]*Card, 0)
	actionCards := make([]*Card, 0)
	modifierCards := make([]*Card, 0)

	for _, card := range d.cards {
		switch card.Type {
		case NumberCard:
			numberCards = append(numberCards, card)
		case ActionCard:
			actionCards = append(actionCards, card)
		case ModifierCard:
			modifierCards = append(modifierCards, card)
		}
	}

	// Display cards by category
	cardOptions := make([]*Card, 0)
	optionIndex := 1

	if len(numberCards) > 0 {
		fmt.Println("\nNumber Cards:")
		cardCounts := make(map[int]int)
		for _, card := range numberCards {
			cardCounts[card.Value]++
		}
		for value := 0; value <= 12; value++ {
			if count := cardCounts[value]; count > 0 {
				fmt.Printf("  %d) [%d] (%d available)\n", optionIndex, value, count)
				cardOptions = append(cardOptions, NewNumberCard(value))
				optionIndex++
			}
		}
	}

	if len(actionCards) > 0 {
		fmt.Println("\nAction Cards:")
		actionCounts := make(map[ActionType]int)
		for _, card := range actionCards {
			actionCounts[card.Action]++
		}
		actionNames := []string{"❄️ FREEZE", "🎲 FLIP 3", "🆘 2ND CHANCE"}
		for i, count := range []int{actionCounts[Freeze], actionCounts[FlipThree], actionCounts[SecondChance]} {
			if count > 0 {
				fmt.Printf("  %d) %s (%d available)\n", optionIndex, actionNames[i], count)
				cardOptions = append(cardOptions, NewActionCard(ActionType(i)))
				optionIndex++
			}
		}
	}

	if len(modifierCards) > 0 {
		fmt.Println("\nModifier Cards:")
		modifierCounts := make(map[ModifierType]int)
		for _, card := range modifierCards {
			modifierCounts[card.Modifier]++
		}
		modifierNames := []string{"+2", "+4", "+6", "+8", "+10", "×2"}
		for i, count := range []int{modifierCounts[Plus2], modifierCounts[Plus4], modifierCounts[Plus6], modifierCounts[Plus8], modifierCounts[Plus10], modifierCounts[Multiply2]} {
			if count > 0 {
				fmt.Printf("  %d) [%s] (%d available)\n", optionIndex, modifierNames[i], count)
				cardOptions = append(cardOptions, NewModifierCard(ModifierType(i)))
				optionIndex++
			}
		}
	}

	fmt.Printf("\nEnter choice (1-%d): ", len(cardOptions))

	for {
		if !d.scanner.Scan() {
			// Fall back to random card if input fails
			return d.drawRandomCard()
		}

		input := strings.TrimSpace(d.scanner.Text())
		choice, err := strconv.Atoi(input)
		if err != nil || choice < 1 || choice > len(cardOptions) {
			fmt.Printf("Please enter a number between 1 and %d: ", len(cardOptions))
			continue
		}

		selectedCard := cardOptions[choice-1]

		// Find and remove the actual card from the deck
		for i, card := range d.cards {
			if d.cardsEqual(card, selectedCard) {
				// Remove this card from the deck
				d.cards = append(d.cards[:i], d.cards[i+1:]...)
				return card
			}
		}

		// Fallback if card not found (shouldn't happen)
		fmt.Println("Card not found, drawing random card instead...")
		return d.drawRandomCard()
	}
}

// drawRandomCard draws a random card (fallback method)
func (d *Deck) drawRandomCard() *Card {
	if len(d.cards) == 0 {
		return nil
	}

	card := d.cards[len(d.cards)-1]
	d.cards = d.cards[:len(d.cards)-1]
	return card
}

// cardsEqual checks if two cards are equivalent
func (d *Deck) cardsEqual(card1, card2 *Card) bool {
	if card1.Type != card2.Type {
		return false
	}

	switch card1.Type {
	case NumberCard:
		return card1.Value == card2.Value
	case ActionCard:
		return card1.Action == card2.Action
	case ModifierCard:
		return card1.Modifier == card2.Modifier
	}

	return false
}
