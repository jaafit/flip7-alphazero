# 🎴 Flip 7 - CLI Edition

A command-line implementation of the exciting press-your-luck card game **Flip 7** by Eric Olsen. Push your luck, collect cards, and be the first to reach 200 points!

## 🎯 Game Objective

Be the first player to score **200 or more points** by collecting cards across multiple rounds. But beware - draw duplicate number cards and you'll bust, losing all points for that round!

## 🃏 How to Play

### Basic Rules
- Each round, players are dealt cards one at a time, face up
- On your turn, choose to:
  - **Hit** - Draw another card (risk vs. reward)
  - **Stay** - Bank your current points and exit the round
- **Bust** if you draw a duplicate number card (lose all round points)
- **Flip 7** if you collect 7 different number cards (15 bonus points + round ends)

### Card Types

#### 🔢 Number Cards (0-12)
- Worth their face value in points
- The deck contains as many of each number as the number itself
  - 12 copies of "12", 11 copies of "11", down to 1 copy of "0"

#### ⚡ Action Cards
- **❄️ Freeze** - Forces a player to stay and bank their points
- **🎲 Flip Three** - Forces a player to draw 3 cards in succession  
- **🆘 Second Chance** - Automatically goes to the player who drew it (unless they already have one), allows avoiding a bust once by discarding a duplicate

#### 🎯 Modifier Cards
- **+2, +4, +6, +8, +10** - Add points to your number card total
- **×2** - Doubles your number card total (before adding other modifiers)

### Scoring
1. Add up your number cards
2. Apply ×2 multiplier if you have one
3. Add any +point modifier cards
4. Add 15 bonus points for Flip 7

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher (no external dependencies)

### Run the game
```bash
# Clone or download the project
git clone <repository-url>
cd flip7-alphazero

# Run the game
python3 main.py

# Run with debug mode (choose cards manually)
python3 main.py --debug
```

### Quick Start
```bash
# Normal game
python3 main.py

# Debug mode
python3 main.py --debug
```

### Game Modes
- **Mixed Mode**: Human and computer players together
- **AI-Only Mode**: Watch computer players battle (0 human players)
- **Debug Mode**: Manually choose every card drawn

## 🎮 Gameplay Example

```
🎴 Welcome to Flip 7!
Press your luck and flip your way to 200 points!

How many players total? (2-18): 3
Enter name for Player 1: Alice
Enter name for Player 2: Bob  
Enter name for Player 3: Charlie

🎮 Starting Flip 7! First to 200 points wins!

==================================================
🎯 ROUND 1
==================================================
Dealer: Alice

🃏 Dealing initial cards...
   Bob draws [7]
   Charlie draws [9]
   Alice draws [10]

👤 Bob:
   Numbers: [7]

🎯 Bob, do you want to (H)it or (S)tay? h
   Bob draws [×2]
👤 Bob:
   Numbers: [7]
   Modifiers: [×2]

🎯 Bob, do you want to (H)it or (S)tay? h
   Bob draws [3]
👤 Bob:
   Numbers: [7] [3]
   Modifiers: [×2]

🎯 Bob, do you want to (H)it or (S)tay? s
   Bob stays with 20 points

📊 Calculating round scores...
----------------------------------------
Bob: 20 points this round (Total: 20)
Charlie: 9 points this round (Total: 9)
Alice: 10 points this round (Total: 10)
----------------------------------------
```

## 🎲 Strategy Tips

1. **Know the odds** - Higher numbers have more copies, increasing bust risk
2. **The zero card** is safe - only one copy in the deck, can't cause a bust
3. **Second Chance cards** are powerful - they automatically go to whoever draws them (unless they already have one), so save them for the right moment
4. **Risk vs. reward** - Push your luck for higher scores, but know when to stay
5. **Modifier cards** can multiply your score - but you need number cards first

## 🏗️ Project Structure

```
flip7-alphazero/
├── main.py           # Entry point
├── game.py           # Main game logic and flow
├── card.py           # Card types and definitions
├── deck.py           # Deck management and shuffling
├── player.py         # Player state and hand management
├── human_player.py   # Human player (stdin I/O)
├── computer_player.py # Computer players and AI strategies
├── requirements.txt  # Dependencies (stdlib only)
└── README.md         # This file
```

## 🛠️ Development

### Code Organization
- **Card System**: Flexible card types with proper game logic
- **Player Management**: State tracking, scoring, and hand display
- **Game Engine**: Turn management, action card handling, and round flow
- **CLI Interface**: User input validation and game state display

### Debug Mode
The game includes a debug mode for testing and development:

```bash
./flip7 -debug
```

In debug mode, you can manually choose which card to draw each turn:
- Cards are organized by type (Number, Action, Modifier)
- Shows how many of each card type are available
- Perfect for testing specific scenarios like Flip 7, action card interactions, etc.

**Example debug card selection:**
```
🐛 DEBUG: Choose a card to draw:
Available cards (94 total):

Number Cards:
  1) [0] (1 available)
  2) [1] (1 available)
  3) [2] (2 available)
  ...

Action Cards:
  14) ❄️ FREEZE (3 available)
  15) 🎲 FLIP 3 (3 available)
  16) 🆘 2ND CHANCE (3 available)

Modifier Cards:
  17) [+2] (1 available)
  18) [+4] (1 available)
  ...

Enter choice (1-22): 
```

## 🤖 Computer Players

The game supports computer players with different AI strategies:

### AI Strategies
- **Conservative**: Plays it safe, stays early with lower scores
- **Aggressive**: Takes big risks, pushes for higher scores
- **Adaptive**: Adjusts strategy based on game state and other players
- **Chaotic**: Unpredictable decisions for fun and variety

### AI-Only Mode
You can create games with zero human players to:
- Test different AI strategies against each other
- Watch entertaining computer battles
- Study optimal play patterns
- Let the game run hands-free

**Example AI-only setup:**
```
How many players total? (2-8): 4
How many human players? (0-4): 0

Computer Player 1: Conservative → "Cautious Carl"
Computer Player 2: Aggressive → "Bold Bob" 
Computer Player 3: Adaptive → "Smart Sam"
Computer Player 4: Chaotic → "Wild Will"

🎮 Starting AI-only Flip 7 with 4 computer players!
🍿 Sit back and watch the AIs battle it out!
```

### Key Features Implemented
- ✅ Full Flip 7 rules implementation
- ✅ All 3 action card types with proper interactions
- ✅ Modifier cards with correct scoring
- ✅ Second Chance mechanics
- ✅ Deck management with reshuffling
- ✅ Input validation and error handling
- ✅ Visual game state display
- ✅ Winner detection
- ✅ Computer players with AI strategies
- ✅ AI-only mode (watch computers play)
- ✅ Debug mode for manual card selection

## 📄 License

This is an implementation of the card game **Flip 7** designed by Eric Olsen and published by The Op Games (2024). This CLI version is created for educational and entertainment purposes.

## 🎉 Credits

- **Original Game**: Flip 7 by Eric Olsen (The Op Games, 2024)
- **CLI Implementation**: Built with Python
- **Inspiration**: Love for press-your-luck games and terminal interfaces

---

**Have fun flipping your way to victory!** 🎴✨ 