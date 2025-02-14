import math
import copy
import time
import argparse
import sys

class MiniChess:
    def __init__(self):
        self.current_game_state = self.init_board()
        self.no_capture_turns = 0
        self.turnNumber = 0
        self.output_file = open("minichessoutput.txt", "w")

    def __del__(self):
        if hasattr(self, 'output_file'):
            self.output_file.close()

    """
    Initialize the board

    Args:
        - None
    Returns:
        - state: A dictionary representing the state of the game
    """
    def init_board(self):
        state = {
            "board": 
            [['bK', 'bQ', 'bB', 'bN', '.'],
            ['.', '.', 'bp', 'bp', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', 'wp', 'wp', '.', '.'],
            ['.', 'wN', 'wB', 'wQ', 'wK']],
            "turn": 'white',
        }
        return state

    """
    Prints the board
    
    Args:
        - game_state: Dictionary representing the current game state
    Returns:
        - None
    """
    def display_board(self, game_state):
        board_output = "\n"
        for i, row in enumerate(game_state["board"], start=1):
            board_output += str(6-i) + "  " + ' '.join(piece.rjust(3) for piece in row) + "\n"
        board_output += "\n     A   B   C   D   E\n"
        self.write_to_file(board_output)

    """
    Writes text to both console and output file

    Args:
        - text: string | The text to write
    Returns:
        - None
    """
    def write_to_file(self, text):
        print(text)  # Print to console
        self.output_file.write(text + "\n")  # Write to file

    """
    Check if the move is valid    
    
    Args: 
        - game_state: dictionary | Dictionary representing the current game state
        - move: tuple | The move which we check the validity of ((start_row, start_col), (end_row, end_col))
    Returns:
        - boolean representing the validity of the move
    """
    def is_valid_move(self, game_state, move):
        start, end = move
        start_row, start_col = start
        end_row, end_col = end
        board = game_state["board"]

        # Check if start and end positions are within bounds
        if not (0 <= start_row < 5 and 0 <= start_col < 5 and 0 <= end_row < 5 and 0 <= end_col < 5):
            return False

        piece = board[start_row][start_col]

        # Check if the starting position is empty
        if piece == '.':
            return False

        # Check if it's the correct player's turn
        if (game_state["turn"] == "white" and not piece.startswith('w')) or \
           (game_state["turn"] == "black" and not piece.startswith('b')):
            return False

        # Prevent moving onto your own piece
        if board[end_row][end_col] != '.' and board[end_row][end_col][0] == piece[0]:
            return False

        # Implement movement rules for each piece (simplified for this example)
        return True

    """
    Returns a list of valid moves

    Args:
        - game_state: dictionary | Dictionary representing the current game state
    Returns:
        - valid_moves: list | A list of nested tuples corresponding to valid moves [((start_row, start_col), (end_row, end_col))]
    """
    def valid_moves(self, game_state):
        moves = []
        for row in range(5):
            for col in range(5):
                piece = game_state["board"][row][col]
                if (game_state["turn"] == "white" and piece.startswith('w')) or \
                   (game_state["turn"] == "black" and piece.startswith('b')):
                    for r in range(5):
                        for c in range(5):
                            move = ((row, col), (r, c))
                            if self.is_valid_move(game_state, move):
                                moves.append(move)
        return moves

    """
    Modify to board to make a move

    Args: 
        - game_state: dictionary | Dictionary representing the current game state
        - move: tuple | The move to perform ((start_row, start_col), (end_row, end_col))
    Returns:
        - game_state: dictionary | Dictionary representing the modified game state
    """
    def make_move(self, game_state, move):
        start = move[0]
        end = move[1]
        start_row, start_col = start
        end_row, end_col = end
        piece = game_state["board"][start_row][start_col]
        
        # Perform move on the board
        game_state["board"][start_row][start_col] = '.'
        game_state["board"][end_row][end_col] = piece

        # Pawn promotion logic (if applicable)
        if piece[1] == 'p' and (end_row == 0 or end_row == 4):
            game_state["board"][end_row][end_col] = piece[0] + 'Q'

        # Switch turn after making a move
        game_state["turn"] = "black" if game_state["turn"] == "white" else "white"

        return game_state

    """
    Parse the input string and modify it into board coordinates

    Args:
        - move: string | String representing a move "B2 B3"
    Returns:
        - tuple | The parsed move as ((start_row, start_col), (end_row, end_col))
    """
    def parse_input(self, move):
        try:
            start, end = move.split()
            start = (5-int(start[1]), ord(start[0].upper()) - ord('A'))
            end = (5-int(end[1]), ord(end[0].upper()) - ord('A'))
            return (start, end)
        except:
            return None

    """
    Game loop
    
    Args:
       - None
    Returns:
       - None
    """
    def play(self):
        self.write_to_file("Welcome to Mini Chess! Enter moves as 'B2 B3'. Type 'exit' to quit.")
        
        while True:
            self.display_board(self.current_game_state)
            
            # Check for game over conditions
            flat_board = [piece for row in self.current_game_state["board"] for piece in row]
            if 'bK' not in flat_board:
                self.write_to_file(f"White wins after {self.turnNumber} turns!")
                break
            elif 'wK' not in flat_board:
                self.write_to_file(f"Black wins after {self.turnNumber} turns!")
                break
            
            if self.no_capture_turns >= 20:
                self.write_to_file("Draw! No captures in the last 10 turns.")
                break
            
            self.write_to_file(f"{self.no_capture_turns} turns since last capture.")
            
            # Get player input
            move = input(f"{self.current_game_state['turn'].capitalize()} to move: ")
            self.write_to_file(f"{self.current_game_state['turn'].capitalize()} to move: {move}")
            
            if move.lower() == 'exit':
                self.write_to_file("Game exited.")
                sys.exit(1)

            parsed_move = self.parse_input(move)
            
            if not parsed_move or not self.is_valid_move(self.current_game_state, parsed_move):
                self.write_to_file("Invalid move. Try again.")
                continue
            
            # Check if a capture occurs during the move
            _, dest = parsed_move
            dest_piece = self.current_game_state["board"][dest[0]][dest[1]]
            
            # Make the move on the board
            self.make_move(self.current_game_state, parsed_move)
            
            # Update capture counter and turn number
            self.no_capture_turns = self.no_capture_turns + 1 if dest_piece == '.' else 0
            self.turnNumber += 1


if __name__ == "__main__":
    game = MiniChess()
    game.play()
