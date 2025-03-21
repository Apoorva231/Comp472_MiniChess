import math
import copy
import time
import argparse
import sys


class MiniChess:
    def __init__(self, timeout, max_turns, use_alpha_beta, play_mode, heuristic):
        self.current_game_state = self.init_board()
        self.no_capture_turns = 0
        self.turnNumber = 0
        self.timeout = timeout
        self.max_turns = max_turns
        self.use_alpha_beta = use_alpha_beta
        self.play_mode = play_mode
        self.heuristic = heuristic

        # AI statistics
        self.states_explored = 0
        self.states_by_depth = {}
        self.total_branching = 0
        self.total_decisions = 0

        # Setup output file
        alpha_beta_str = "true" if use_alpha_beta else "false"
        output_filename = f"gameTrace-{alpha_beta_str}-{timeout}-{max_turns}.txt"
        self.output_file = open(output_filename, "w")

        # Write game parameters to the trace file
        self.write_game_parameters()

    def __del__(self):
        if hasattr(self, 'output_file'):
            self.output_file.close()

    def write_to_file(self, text):
        print(text)  # Print to console
        self.output_file.write(text + "\n")  # Write to file

    def write_game_parameters(self):
        self.write_to_file("===== GAME PARAMETERS =====")
        self.write_to_file(f"Timeout: {self.timeout} seconds")
        self.write_to_file(f"Max turns: {self.max_turns}")
        self.write_to_file(f"Play mode: {self.play_mode}")

        if "AI" in self.play_mode:
            self.write_to_file(f"Alpha-beta: {'ON' if self.use_alpha_beta else 'OFF'}")
            self.write_to_file(f"Heuristic: {self.heuristic}")

        self.write_to_file("===========================")

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
            board_output += str(6 - i) + "  " + ' '.join(piece.rjust(3) for piece in row) + "\n"
        board_output += "\n     A   B   C   D   E\n"
        self.write_to_file(board_output)

    """
    Check if the move is valid    

    Args: 
        - game_state:   dictionary | Dictionary representing the current game state
        - move          tuple | the move which we check the validity of ((start_row, start_col),(end_row, end_col))
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

        # Implement movement rules for each piece
        if piece[1] == 'K':  # King
            return abs(end_row - start_row) <= 1 and abs(end_col - start_col) <= 1
        elif piece[1] == 'Q':  # Queen
            return self.is_straight_or_diagonal_move(start, end, board)
        elif piece[1] == 'B':  # Bishop
            return self.is_diagonal_move(start, end, board)
        elif piece[1] == 'N':  # Knight
            return (abs(end_row - start_row), abs(end_col - start_col)) in [(2, 1), (1, 2)]
        elif piece[1] == 'p':  # Pawn
            direction = -1 if piece.startswith('w') else 1
            if end_col == start_col:  # Move forward
                return end_row == start_row + direction and board[end_row][end_col] == '.'
            elif abs(end_col - start_col) == 1:  # Capture diagonally
                return end_row == start_row + direction and board[end_row][end_col].startswith(
                    'b' if piece.startswith('w') else 'w')

        return False

    def is_straight_or_diagonal_move(self, start, end, board):
        start_row, start_col = start
        end_row, end_col = end

        delta_row = end_row - start_row
        delta_col = end_col - start_col

        if delta_row == 0 or delta_col == 0:  # Straight line
            step_row = 0 if delta_row == 0 else delta_row // abs(delta_row)
            step_col = 0 if delta_col == 0 else delta_col // abs(delta_col)
        elif abs(delta_row) == abs(delta_col):  # Diagonal line
            step_row = delta_row // abs(delta_row)
            step_col = delta_col // abs(delta_col)
        else:
            return False

        current_row, current_col = start_row + step_row, start_col + step_col
        while (current_row != end_row or current_col != end_col):
            if board[current_row][current_col] != '.':
                return False
            current_row += step_row
            current_col += step_col

        return True

    def is_diagonal_move(self, start, end, board):
        delta_row = abs(end[0] - start[0])
        delta_col = abs(end[1] - start[1])

        return delta_row == delta_col and self.is_straight_or_diagonal_move(start, end, board)

    """
    Returns a list of valid moves

    Args:
        - game_state:   dictionary | Dictionary representing the current game state
    Returns:
        - valid moves:   list | A list of nested tuples corresponding to valid moves [((start_row, start_col),(end_row, end_col)),((start_row, start_col),(end_row, end_col))]
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
        - game_state:   dictionary | Dictionary representing the current game state
        - move          tuple | the move to perform ((start_row, start_col),(end_row, end_col))
    Returns:
        - game_state:   dictionary | Dictionary representing the modified game state
    """

    def make_move(self, game_state, move):
        start, end = move
        start_row, start_col = start
        end_row, end_col = end
        piece = game_state["board"][start_row][start_col]
        game_state["board"][start_row][start_col] = '.'
        game_state["board"][end_row][end_col] = piece

        # Pawn promotion
        if piece[1] == 'p' and (end_row == 0 or end_row == 4):
            game_state["board"][end_row][end_col] = piece[0] + 'Q'

        game_state["turn"] = "black" if game_state["turn"] == "white" else "white"

        return game_state

    """
    Parse the input string and modify it into board coordinates

    Args:
        - move: string representing a move "B2 B3"
    Returns:
        - (start, end)  tuple | the move to perform ((start_row, start_col),(end_row, end_col))
    """

    def parse_input(self, move):
        try:
            start, end = move.split()
            start = (5 - int(start[1]), ord(start[0].upper()) - ord('A'))
            end = (5 - int(end[1]), ord(end[0].upper()) - ord('A'))
            return (start, end)
        except:
            return None

    """
    Convert board coordinates to algebraic notation

    Args:
        - move: tuple | the move to convert ((start_row, start_col),(end_row, end_col))
    Returns:
        - string representing a move "B2 B3"
    """

    def move_to_algebraic(self, move):
        start, end = move
        start_row, start_col = start
        end_row, end_col = end

        start_algebraic = chr(start_col + ord('A')) + str(5 - start_row)
        end_algebraic = chr(end_col + ord('A')) + str(5 - end_row)

        return f"{start_algebraic} {end_algebraic}"

    """
    Calculate the heuristic value of a game state

    Args:
        - game_state: dictionary | Dictionary representing the current game state
        - heuristic: string | The heuristic to use ('e0', 'e1', 'e2')
    Returns:
        - float | The heuristic value of the game state
    """

    def evaluate(self, game_state, heuristic='e0'):
        board = game_state["board"]

        # Count pieces
        piece_counts = {
            'wp': 0, 'wB': 0, 'wN': 0, 'wQ': 0, 'wK': 0,
            'bp': 0, 'bB': 0, 'bN': 0, 'bQ': 0, 'bK': 0
        }

        for row in board:
            for piece in row:
                if piece != '.':
                    piece_counts[piece] += 1

        if heuristic == 'e0':
            # e0 = (#wp + 3 · #wB + 3 · #wN + 9 · #wQ + 999 · wK)
            # −(#bp + 3 · #bB + 3 · #bN + 9 · #bQ + 999 · bK)
            white_score = (piece_counts['wp'] +
                           3 * piece_counts['wB'] +
                           3 * piece_counts['wN'] +
                           9 * piece_counts['wQ'] +
                           999 * piece_counts['wK'])

            black_score = (piece_counts['bp'] +
                           3 * piece_counts['bB'] +
                           3 * piece_counts['bN'] +
                           9 * piece_counts['bQ'] +
                           999 * piece_counts['bK'])

            return white_score - black_score

        elif heuristic == 'e1':
            # e1: Similar to e0 but with additional positional considerations
            # Center control bonus for knights and bishops
            white_score = (piece_counts['wp'] +
                           3 * piece_counts['wB'] +
                           3 * piece_counts['wN'] +
                           9 * piece_counts['wQ'] +
                           999 * piece_counts['wK'])

            black_score = (piece_counts['bp'] +
                           3 * piece_counts['bB'] +
                           3 * piece_counts['bN'] +
                           9 * piece_counts['bQ'] +
                           999 * piece_counts['bK'])

            # Add positional bonuses
            for r in range(5):
                for c in range(5):
                    piece = board[r][c]
                    if piece == '.':
                        continue

                    # Center control bonus for knights and bishops
                    if piece in ['wN', 'wB'] and 1 <= r <= 3 and 1 <= c <= 3:
                        white_score += 0.5
                    elif piece in ['bN', 'bB'] and 1 <= r <= 3 and 1 <= c <= 3:
                        black_score += 0.5

                    # Pawn advancement bonus
                    if piece == 'wp':
                        white_score += 0.1 * (4 - r)  # More advanced pawns get higher bonus
                    elif piece == 'bp':
                        black_score += 0.1 * r  # More advanced pawns get higher bonus

            return white_score - black_score

        elif heuristic == 'e2':
            # e2: Similar to e0 but with mobility considerations
            white_score = (piece_counts['wp'] +
                           3 * piece_counts['wB'] +
                           3 * piece_counts['wN'] +
                           9 * piece_counts['wQ'] +
                           999 * piece_counts['wK'])

            black_score = (piece_counts['bp'] +
                           3 * piece_counts['bB'] +
                           3 * piece_counts['bN'] +
                           9 * piece_counts['bQ'] +
                           999 * piece_counts['bK'])

            # Calculate mobility (number of legal moves)
            original_turn = game_state["turn"]

            # Check white mobility
            game_state_copy = copy.deepcopy(game_state)
            game_state_copy["turn"] = "white"
            white_moves = len(self.valid_moves(game_state_copy))

            # Check black mobility
            game_state_copy["turn"] = "black"
            black_moves = len(self.valid_moves(game_state_copy))

            # Restore original turn
            game_state["turn"] = original_turn

            # Add mobility factor
            white_score += 0.1 * white_moves
            black_score += 0.1 * black_moves

            return white_score - black_score

        else:
            # Default to e0
            return self.evaluate(game_state, 'e0')

    """
    Minimax algorithm with optional alpha-beta pruning

    Args:
        - game_state: dictionary | Dictionary representing the current game state
        - depth: int | The depth to search to
        - is_maximizing: bool | Whether the current player is maximizing
        - alpha: float | Alpha value for alpha-beta pruning
        - beta: float | Beta value for alpha-beta pruning
        - start_time: float | The time the search started
        - time_limit: float | The maximum time allowed for the search
    Returns:
        - tuple | (best_score, best_move)
    """

    def minimax(self, game_state, depth, is_maximizing, alpha=float('-inf'), beta=float('inf'), start_time=None,
                time_limit=None):
        # Update state exploration statistics
        self.states_explored += 1
        if depth not in self.states_by_depth:
            self.states_by_depth[depth] = 0
        self.states_by_depth[depth] += 1

        # Check for timeout
        if start_time and time_limit and time.time() - start_time > time_limit:
            # Return best move found so far if time limit reached
            return (0 if is_maximizing else 0), None

        # Check for game over
        flat_board = [piece for row in game_state["board"] for piece in row]
        if 'bK' not in flat_board:
            return float('inf'), None  # White wins
        elif 'wK' not in flat_board:
            return float('-inf'), None  # Black wins

        # Base case: reached max depth
        if depth == 0:
            return self.evaluate(game_state, self.heuristic), None

        valid_moves = self.valid_moves(game_state)

        # Update branching factor statistics
        if valid_moves:
            self.total_branching += len(valid_moves)
            if depth == 3:  # Only count top-level decisions for average
                self.total_decisions += 1

        # No valid moves
        if not valid_moves:
            return self.evaluate(game_state, self.heuristic), None

        best_move = None

        if is_maximizing:
            best_score = float('-inf')
            for move in valid_moves:
                # Make a deep copy to avoid modifying the original state
                new_state = copy.deepcopy(game_state)
                self.make_move(new_state, move)

                # Recursive call
                score, _ = self.minimax(new_state, depth - 1, False, alpha, beta, start_time, time_limit)

                if score > best_score:
                    best_score = score
                    best_move = move

                # Alpha-beta pruning
                if self.use_alpha_beta:
                    alpha = max(alpha, best_score)
                    if beta <= alpha:
                        break
        else:
            best_score = float('inf')
            for move in valid_moves:
                # Make a deep copy to avoid modifying the original state
                new_state = copy.deepcopy(game_state)
                self.make_move(new_state, move)

                # Recursive call
                score, _ = self.minimax(new_state, depth - 1, True, alpha, beta, start_time, time_limit)

                if score < best_score:
                    best_score = score
                    best_move = move

                # Alpha-beta pruning
                if self.use_alpha_beta:
                    beta = min(beta, best_score)
                    if beta <= alpha:
                        break

        return best_score, best_move

    """
    Get the best move for the AI

    Args:
        - game_state: dictionary | Dictionary representing the current game state
    Returns:
        - tuple | (best_move, score, heuristic_score, time_taken)
    """

    def get_ai_move(self, game_state):
        start_time = time.time()

        # Initialize variables for iterative deepening
        best_move = None
        best_score = float('-inf') if game_state["turn"] == "white" else float('inf')
        current_depth = 1
        max_depth = 10  # Maximum depth for iterative deepening

        # Reset statistics for this move
        prev_states_explored = self.states_explored

        while current_depth <= max_depth:
            # Check if there's enough time for another iteration
            if time.time() - start_time > self.timeout * 0.8:  # Use 80% of allowed time
                break

            # Run minimax/alpha-beta at the current depth
            is_maximizing = game_state["turn"] == "white"
            score, move = self.minimax(
                game_state,
                current_depth,
                is_maximizing,
                float('-inf'),
                float('inf'),
                start_time,
                self.timeout * 0.95  # Use 95% of allowed time
            )

            # Update best move if we have a complete search at this depth
            if move is not None:
                best_move = move
                best_score = score

            # If we found a winning move, no need to search deeper
            if (is_maximizing and score == float('inf')) or (not is_maximizing and score == float('-inf')):
                break

            current_depth += 1

        end_time = time.time()
        time_taken = end_time - start_time

        # Calculate the heuristic score of the resulting board
        if best_move:
            new_state = copy.deepcopy(game_state)
            self.make_move(new_state, best_move)
            heuristic_score = self.evaluate(new_state, self.heuristic)
        else:
            # If no move found (shouldn't happen), pick the first valid move
            valid_moves = self.valid_moves(game_state)
            if valid_moves:
                best_move = valid_moves[0]
                new_state = copy.deepcopy(game_state)
                self.make_move(new_state, best_move)
                heuristic_score = self.evaluate(new_state, self.heuristic)
                best_score = heuristic_score
            else:
                heuristic_score = self.evaluate(game_state, self.heuristic)

        # Calculate states explored for this move
        states_this_move = self.states_explored - prev_states_explored

        return best_move, best_score, heuristic_score, time_taken, states_this_move, current_depth - 1

    """
    Format a number for display (e.g., 1200 -> 1.2k, 1200000 -> 1.2M)

    Args:
        - num: int | The number to format
    Returns:
        - string | The formatted number
    """

    def format_number(self, num):
        if num < 1000:
            return str(num)
        elif num < 1000000:
            return f"{num / 1000:.1f}k"
        else:
            return f"{num / 1000000:.1f}M"

    """
    Display AI statistics

    Args:
        - states_this_move: int | Number of states explored for this move
        - max_depth: int | Maximum depth reached for this move
    """

    def display_ai_stats(self, states_this_move=0, max_depth=0):
        self.write_to_file("\n===== AI STATISTICS =====")

        # Total states explored
        self.write_to_file(f"Cumulative states explored: {self.format_number(self.states_explored)}")

        # States by depth
        depth_stats = "Cumulative states explored by depth: "
        for depth in sorted(self.states_by_depth.keys()):
            depth_stats += f"{depth}={self.format_number(self.states_by_depth[depth])} "
        self.write_to_file(depth_stats)

        # Percentages by depth
        if self.states_explored > 0:
            percent_stats = "Cumulative % states explored by depth: "
            for depth in sorted(self.states_by_depth.keys()):
                percentage = (self.states_by_depth[depth] / self.states_explored) * 100
                percent_stats += f"{depth}={percentage:.1f}% "
            self.write_to_file(percent_stats)

        # Average branching factor
        if self.total_decisions > 0:
            avg_branching = self.total_branching / self.total_decisions
            self.write_to_file(f"Average branching factor: {avg_branching:.1f}")

        self.write_to_file("==========================\n")

    """
    Game loop

    Args:
        - None
    Returns:
        - None
    """

    def play(self):
        self.write_to_file("Welcome to Mini Chess!")

        # Display initial board
        self.display_board(self.current_game_state)

        while True:
            # Check for game over conditions
            flat_board = [piece for row in self.current_game_state["board"] for piece in row]
            if 'bK' not in flat_board:
                self.write_to_file(f"White wins after {self.turnNumber} turns!")
                break
            elif 'wK' not in flat_board:
                self.write_to_file(f"Black wins after {self.turnNumber} turns!")
                break

            if self.no_capture_turns >= self.max_turns:
                self.write_to_file(f"Draw! No captures in the last {self.max_turns} turns.")
                break

            if self.turnNumber >= self.max_turns * 2:  # Both players get max_turns each
                self.write_to_file(f"Draw! Maximum number of turns ({self.max_turns}) reached.")
                break

            current_player = self.current_game_state["turn"]
            self.write_to_file(f"Turn {self.turnNumber}.")
            self.write_to_file(f"{self.no_capture_turns} turns since last capture.")

            # Determine if current player is AI
            is_ai = False
            if (current_player == "white" and self.play_mode in ["AI-H", "AI-AI"]) or \
                    (current_player == "black" and self.play_mode in ["H-AI", "AI-AI"]):
                is_ai = True

            # Get move based on player type
            if is_ai:
                self.write_to_file(f"{current_player.capitalize()} (AI) is thinking...")

                # Get AI move
                move, search_score, heuristic_score, time_taken, states_this_move, max_depth = self.get_ai_move(
                    self.current_game_state)

                if move:
                    move_str = self.move_to_algebraic(move)
                    self.write_to_file(f"{current_player.capitalize()} (AI) move: {move_str}")
                    self.write_to_file(f"Time for this action: {time_taken:.2f} sec")
                    self.write_to_file(f"Heuristic score: {heuristic_score}")
                    self.write_to_file(
                        f"{'Alpha-beta' if self.use_alpha_beta else 'Minimax'} search score: {search_score}")

                    # Display AI statistics
                    self.display_ai_stats(states_this_move, max_depth)
                else:
                    self.write_to_file(f"{current_player.capitalize()} (AI) could not find a valid move!")
                    if current_player == "white":
                        self.write_to_file("Black wins due to AI error!")
                    else:
                        self.write_to_file("White wins due to AI error!")
                    break
            else:
                # Human move
                move_str = input(f"{current_player.capitalize()} to move: ")
                self.write_to_file(f"{current_player.capitalize()} to move: {move_str}")

                if move_str.lower() == 'exit':
                    self.write_to_file("Game exited.")
                    sys.exit(1)

                move = self.parse_input(move_str)

            # Validate and make the move
            if not move or not self.is_valid_move(self.current_game_state, move):
                if is_ai:
                    # AI made an invalid move - it loses
                    self.write_to_file(
                        f"Invalid move by {current_player} AI: {self.move_to_algebraic(move) if move else 'None'}")
                    if current_player == "white":
                        self.write_to_file("Black wins due to invalid AI move!")
                    else:
                        self.write_to_file("White wins due to invalid AI move!")
                    break
                else:
                    # Human made an invalid move - try again
                    self.write_to_file("Invalid move. Try again.")
                    continue

            # Check if a capture occurs
            _, dest = move
            dest_piece = self.current_game_state["board"][dest[0]][dest[1]]
            is_capture = dest_piece != '.'

            # Make the move
            self.make_move(self.current_game_state, move)

            # Update capture counter
            self.no_capture_turns = 0 if is_capture else self.no_capture_turns + 1
            self.turnNumber += 1

            # Display the updated board
            self.display_board(self.current_game_state)


def parse_arguments(aiTime, alphaBeta, gameMode, heuristicOption):
    parser = argparse.ArgumentParser(description='MiniChess Game with AI')
    parser.add_argument('-t', '--timeout', type=float, default=aiTime,
                        help='Maximum time (in seconds) for AI to make a move')
    parser.add_argument('-m', '--max_turns', type=int, default=20,
                        help='Maximum number of turns without capture before declaring a draw')
    parser.add_argument('-a', '--alpha_beta', action='store_true', default=alphaBeta,
                        help='Use alpha-beta pruning (default: True)')
    parser.add_argument('-p', '--play_mode', type=str, default=gameMode, choices=['H-H', 'H-AI', 'AI-H', 'AI-AI'],
                        help='Play mode: Human-Human, Human-AI, AI-Human, or AI-AI')
    parser.add_argument('-e', '--heuristic', type=str, default=heuristicOption, choices=['e0', 'e1', 'e2'],
                        help='Heuristic to use for AI (default: e0)')

    return parser.parse_args()


if __name__ == "__main__":

    aiTime=-1;
    while aiTime<=0:
        aiTime=float(input("Enter the maximum time for AI to make a new move: "))

    maxTurns=20

    alphaBeta=""
    while alphaBeta!="yes" and alphaBeta!="no":
        alphaBeta=input("Type 'yes' for alpha beta or 'no' for mini-max: ").lower()

    if alphaBeta == 'yes':
        alphaBeta = True
    else:
        alphaBeta = False

    gameMode=""
    while gameMode not in ["H-H", "H-AI", "AI-H", "AI-AI"]:
        gameMode=input("Select mode option: H-H, H-AI, AI-H, AI-AI: ").upper()

    heuristicOption=""
    while heuristicOption not in ["e0", "e1", "e2"]:
        heuristicOption=input("Select heuristic option: e0, e1, e2: ").lower()

    #args = parse_arguments(aiTime, alphaBeta, gameMode, heuristicOption)
    game = MiniChess(
        aiTime, maxTurns,alphaBeta, gameMode, heuristicOption
    )
    game.play()