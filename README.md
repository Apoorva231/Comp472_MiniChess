# Comp472 MiniChess

MiniChess is a simplified 5x5 chess game for COMP 472. It supports human-vs-human, human-vs-AI, AI-vs-human, and AI-vs-AI games. The AI uses minimax search with optional alpha-beta pruning and one of three heuristic evaluation functions.

## Project Structure

```text
.
├── minichess.py          # Main entry point
└── mini_chess/
    ├── ai.py            # Heuristics, minimax, alpha-beta pruning, AI statistics
    ├── cli.py           # Command-line prompts and arguments
    ├── game.py          # Game loop and trace-file output
    └── rules.py         # Board setup, move parsing, legal moves, piece movement
```

The old experimental files `minichess1.py` and `test472.py` were consolidated into this structure. `minichess1.py` was the working version because it included the AI modes, heuristics, trace output, and alpha-beta option. `test472.py` was an older human-only prototype.

## How to Run

From the repository root:

```bash
python3 minichess.py
```

The program will prompt for:

- AI thinking time
- Alpha-beta pruning or plain minimax
- Play mode: `H-H`, `H-AI`, `AI-H`, or `AI-AI`
- Heuristic: `e0`, `e1`, or `e2`

You can also provide options directly:

```bash
python3 minichess.py --timeout 1 --alpha-beta --play-mode AI-AI --heuristic e0
```

Use plain minimax instead of alpha-beta:

```bash
python3 minichess.py --timeout 1 --minimax --play-mode H-AI --heuristic e1
```

## Move Format

Human moves use algebraic square notation:

```text
B2 B3
```

Type `exit` during a human turn to end the game.

## Game Rules

MiniChess is played on a 5x5 board with files `A-E` and ranks `1-5`.

Initial position:

```text
5   bK  bQ  bB  bN   .
4    .   .  bp  bp   .
3    .   .   .   .   .
2    .  wp  wp   .   .
1    .  wN  wB  wQ  wK

     A   B   C   D   E
```

White moves first. Players alternate turns, and a move is legal only when the selected piece belongs to the player whose turn it is. A piece may not move onto a square occupied by another piece of the same color.

Piece movement:

- King: moves one square in any direction.
- Queen: moves any number of clear squares horizontally, vertically, or diagonally.
- Bishop: moves any number of clear squares diagonally.
- Knight: moves in an `L` shape: two squares in one direction and one square perpendicular.
- Pawn: moves one square forward into an empty square. White pawns move toward rank `5`; black pawns move toward rank `1`.
- Pawn capture: captures one square diagonally forward.
- Pawn promotion: a pawn promotes to a queen when it reaches the opposite edge of the board.

Win and draw conditions:

- A player wins by capturing the opponent's king.
- The implementation does not enforce check or checkmate; kings are captured directly.
- A draw is declared after the configured no-capture turn limit.
- A draw is also declared after both players reach the configured maximum turn limit.

## Game Output

Each run creates a trace file in the current working directory:

```text
gameTrace-{alpha_beta}-{timeout}-{max_turns}.txt
```

For example:

```text
gameTrace-true-1.0-20.txt
```

The trace includes the selected game parameters, board states, moves, heuristic scores, search scores, timing, and cumulative AI statistics.

## Heuristics

- `e0`: Material count using pawn, bishop, knight, queen, and king values.
- `e1`: Material count plus center-control bonuses and pawn-advancement bonuses.
- `e2`: Material count plus legal-move mobility.

## Notes

- Trace files are ignored by git through `.gitignore`.
