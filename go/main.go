package main

import (
	"flag"
	"fmt"
	"os"
)

var debugMode = flag.Bool("debug", false, "Enable debug mode to manually choose cards")

func main() {
	flag.Parse()

	fmt.Println("🎴 Welcome to Flip 7!")
	fmt.Println("Press your luck and flip your way to 200 points!")
	if *debugMode {
		fmt.Println("🐛 DEBUG MODE: You can choose cards manually!")
	}
	fmt.Println()

	game := NewGame()
	game.SetDebugMode(*debugMode)
	if err := game.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
