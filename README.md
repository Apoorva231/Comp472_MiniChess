# Comp472_MiniChess
MiniChess game for Comp472

Mini Chess is a simplified version of chess played on a 5x5 board. This implementation allows two players to play against each other, following specific rules for piece movement and game end conditions.The player can toggle between the following play modes:
human-human, human-ai, ai-human or ai-ai. 


## How to Run
1. Ensure you have Python 3.x installed on your system.
2. Download test.py file
3. Open a terminal/command prompt and navigate to the directory containing `minichess1.py`.
4. Run the command: `python minichess1.py`
5. Follow the on-screen prompts to play the game.

## Game Output
The game generates an output file named 'gameTrace-{alpha_beta_str}-{timeout}-{max_turns}.txt' in the same directory where alpha_beta_str is either True or False, timeout is the ai thinking time and max_turns is turns before the game results in a draw. This file contains a log of all moves and board states throughout the game.

## New Classes/Functions
- `MiniChess`: Main class containing all game logic
  - `write_to_file(text)`: Writes text to both console and output file
  - `display_board(game_state)`: Displays the current board state
  - `is_valid_move(game_state, move)`: Checks if a move is valid
  - `is_straight_or_diagonal_move(start, end, board)`: Checks if a move is a valid straight or diagonal move
  - `is_diagonal_move(start, end, board)`: Checks if a move is a valid diagonal move
  - `make_move(game_state, move)`: Executes a move on the board
  - `parse_input(move)`: Converts user input to board coordinates
  - `play()`: Main game loop
  - `evaluate(self, game_state, heuristic='e0')`: Heuristic evaluation function
  - `minimax(self, game_state, depth, is_maximizing, alpha=float('-inf'), beta=float('inf'), start_time=None,time_limit=None)`: Function for minimax and alpha beta pruning algorithms. 

## Notes
- Enter moves in the format "B2 B3" (from square to square)
- Type 'exit' to quit the game at any time
- `is_straight_or_diagonal_move` and `is_diagonal_move` are helper functions used by `is_valid_move` to check the validity of moves for pieces like the Queen and Bishop

- Link to github: https://github.com/Apoorva231/Comp472_MiniChess 
