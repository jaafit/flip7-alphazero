package main

import (
	"bufio"
	"fmt"
	"strconv"
	"strings"
)

type HumanPlayer struct {
	BasePlayer
	scanner *bufio.Scanner
}

// NewHumanPlayer creates a new human player
func NewHumanPlayer(name string, scanner *bufio.Scanner) *HumanPlayer {
	p := &HumanPlayer{
		scanner: scanner,
	}

	p.BasePlayer.Init(name)

	return p
}

func (p *HumanPlayer) GetPlayerIcon() string {
	return "👤"
}

func (p *HumanPlayer) MakeHitStayDecision(gameState *GameState) (bool, error) {
	fmt.Printf("%s's hand, %v\n", p.Name, p.GetHand())
	fmt.Printf("🎯 %s, do you want to (H)it or (S)tay? ", p.Name)
	for {
		if !p.scanner.Scan() {
			return false, fmt.Errorf("failed to read input")
		}

		choice := strings.ToLower(strings.TrimSpace(p.scanner.Text()))
		if choice == "h" || choice == "hit" {
			return true, nil
		}
		if choice == "s" || choice == "stay" {
			return false, nil
		}

		fmt.Print("Please enter 'H' for Hit or 'S' for Stay: ")
	}
}

func (p *HumanPlayer) ChooseActionTarget(gameState *GameState, actionType ActionType) (PlayerInterface, error) {
	actionName := map[ActionType]string{
		Freeze:       "Who should be frozen?",
		FlipThree:    "Who should flip three cards?",
		SecondChance: "Who should get the Second Chance card?",
	}

	fmt.Printf("   %s\n", actionName[actionType])
	for i, player := range gameState.ActivePlayers {
		fmt.Printf("   %d) %s\n", i+1, player.GetName())
	}

	for {
		fmt.Printf("Enter choice (1-%d): ", len(gameState.ActivePlayers))
		if !p.scanner.Scan() {
			return nil, fmt.Errorf("failed to read input")
		}

		input := strings.TrimSpace(p.scanner.Text())
		choice, err := strconv.Atoi(input)
		if err != nil || choice < 1 || choice > len(gameState.ActivePlayers) {
			fmt.Printf("Please enter a number between 1 and %d: ", len(gameState.ActivePlayers))
			continue
		}

		return gameState.ActivePlayers[choice-1], nil
	}
}

func (p *HumanPlayer) ChoosePositiveActionTarget(gameState *GameState, actionType ActionType) (PlayerInterface, error) {
	actionName := map[ActionType]string{
		Freeze:       "Who should be frozen?",
		FlipThree:    "Who should flip three cards?",
		SecondChance: "Who should get the Second Chance card?",
	}

	fmt.Printf("   %s\n", actionName[actionType])
	for i, player := range gameState.ActivePlayers {
		fmt.Printf("   %d) %s\n", i+1, player.GetName())
	}

	for {
		fmt.Printf("Enter choice (1-%d): ", len(gameState.ActivePlayers))
		if !p.scanner.Scan() {
			return nil, fmt.Errorf("failed to read input")
		}

		input := strings.TrimSpace(p.scanner.Text())
		choice, err := strconv.Atoi(input)
		if err != nil || choice < 1 || choice > len(gameState.ActivePlayers) {
			fmt.Printf("Please enter a number between 1 and %d: ", len(gameState.ActivePlayers))
			continue
		}

		return gameState.ActivePlayers[choice-1], nil
	}
}
