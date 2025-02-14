# Comp472_MiniChess
MiniChess game for Comp472

Mini Chess is a simplified version of chess played on a 5x5 board. This implementation allows two players to play against each other, following specific rules for piece movement and game end conditions.


## How to Run
1. Ensure you have Python 3.x installed on your system.
2. Download test.py file
3. Open a terminal/command prompt and navigate to the directory containing `test.py`.
4. Run the command: `python test.py`
5. Follow the on-screen prompts to play the game.

## Game Output
The game generates an output file named `minichessoutput1.txt` in the same directory. This file contains a log of all moves and board states throughout the game.

## New Classes/Functions
- `MiniChess`: Main class containing all game logic
  - `write_to_file(text)`: Writes text to both console and output file
  - `display_board(game_state)`: Displays the current board state
  - `is_valid_move(game_state, move)`: Checks if a move is valid
  - `make_move(game_state, move)`: Executes a move on the board
  - `parse_input(move)`: Converts user input to board coordinates
  - `play()`: Main game loop

## Notes
- Enter moves in the format "B2 B3" (from square to square)
- Type 'exit' to quit the game at any time
