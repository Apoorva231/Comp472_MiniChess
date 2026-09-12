"""Board setup, move parsing, and MiniChess movement rules."""

from typing import Any, Dict, List, Optional, Tuple

BOARD_SIZE = 5
FILES = "ABCDE"

Board = List[List[str]]
GameState = Dict[str, Any]
Position = Tuple[int, int]
Move = Tuple[Position, Position]

INITIAL_BOARD = [
    ["bK", "bQ", "bB", "bN", "."],
    [".", ".", "bp", "bp", "."],
    [".", ".", ".", ".", "."],
    [".", "wp", "wp", ".", "."],
    [".", "wN", "wB", "wQ", "wK"],
]


def init_board() -> GameState:
    return {
        "board": [row[:] for row in INITIAL_BOARD],
        "turn": "white",
    }


def render_board(game_state: GameState) -> str:
    board_output = "\n"
    for i, row in enumerate(game_state["board"], start=1):
        board_output += str(BOARD_SIZE + 1 - i) + "  " + " ".join(piece.rjust(3) for piece in row) + "\n"
    board_output += "\n     A   B   C   D   E\n"
    return board_output


def is_valid_move(game_state: GameState, move: Move) -> bool:
    start, end = move
    start_row, start_col = start
    end_row, end_col = end
    board = game_state["board"]

    if not (
        0 <= start_row < BOARD_SIZE
        and 0 <= start_col < BOARD_SIZE
        and 0 <= end_row < BOARD_SIZE
        and 0 <= end_col < BOARD_SIZE
    ):
        return False

    piece = board[start_row][start_col]

    if piece == ".":
        return False

    if (game_state["turn"] == "white" and not piece.startswith("w")) or (
        game_state["turn"] == "black" and not piece.startswith("b")
    ):
        return False

    if board[end_row][end_col] != "." and board[end_row][end_col][0] == piece[0]:
        return False

    if piece[1] == "K":
        return abs(end_row - start_row) <= 1 and abs(end_col - start_col) <= 1
    if piece[1] == "Q":
        return is_straight_or_diagonal_move(start, end, board)
    if piece[1] == "B":
        return is_diagonal_move(start, end, board)
    if piece[1] == "N":
        return (abs(end_row - start_row), abs(end_col - start_col)) in [(2, 1), (1, 2)]
    if piece[1] == "p":
        direction = -1 if piece.startswith("w") else 1
        if end_col == start_col:
            return end_row == start_row + direction and board[end_row][end_col] == "."
        if abs(end_col - start_col) == 1:
            return end_row == start_row + direction and board[end_row][end_col].startswith(
                "b" if piece.startswith("w") else "w"
            )

    return False


def is_straight_or_diagonal_move(start: Position, end: Position, board: Board) -> bool:
    start_row, start_col = start
    end_row, end_col = end

    delta_row = end_row - start_row
    delta_col = end_col - start_col

    if delta_row == 0 or delta_col == 0:
        step_row = 0 if delta_row == 0 else delta_row // abs(delta_row)
        step_col = 0 if delta_col == 0 else delta_col // abs(delta_col)
    elif abs(delta_row) == abs(delta_col):
        step_row = delta_row // abs(delta_row)
        step_col = delta_col // abs(delta_col)
    else:
        return False

    current_row, current_col = start_row + step_row, start_col + step_col
    while current_row != end_row or current_col != end_col:
        if board[current_row][current_col] != ".":
            return False
        current_row += step_row
        current_col += step_col

    return True


def is_diagonal_move(start: Position, end: Position, board: Board) -> bool:
    delta_row = abs(end[0] - start[0])
    delta_col = abs(end[1] - start[1])
    return delta_row == delta_col and is_straight_or_diagonal_move(start, end, board)


def valid_moves(game_state: GameState) -> List[Move]:
    moves = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = game_state["board"][row][col]
            if (game_state["turn"] == "white" and piece.startswith("w")) or (
                game_state["turn"] == "black" and piece.startswith("b")
            ):
                for target_row in range(BOARD_SIZE):
                    for target_col in range(BOARD_SIZE):
                        move = ((row, col), (target_row, target_col))
                        if is_valid_move(game_state, move):
                            moves.append(move)
    return moves


def make_move(game_state: GameState, move: Move) -> GameState:
    start, end = move
    start_row, start_col = start
    end_row, end_col = end
    piece = game_state["board"][start_row][start_col]
    game_state["board"][start_row][start_col] = "."
    game_state["board"][end_row][end_col] = piece

    if piece[1] == "p" and (end_row == 0 or end_row == BOARD_SIZE - 1):
        game_state["board"][end_row][end_col] = piece[0] + "Q"

    game_state["turn"] = "black" if game_state["turn"] == "white" else "white"
    return game_state


def parse_input(move: str) -> Optional[Move]:
    try:
        start, end = move.split()
        start_position = (BOARD_SIZE - int(start[1]), FILES.index(start[0].upper()))
        end_position = (BOARD_SIZE - int(end[1]), FILES.index(end[0].upper()))
        return start_position, end_position
    except (ValueError, IndexError):
        return None


def move_to_algebraic(move: Move) -> str:
    start, end = move
    start_row, start_col = start
    end_row, end_col = end

    start_algebraic = chr(start_col + ord("A")) + str(BOARD_SIZE - start_row)
    end_algebraic = chr(end_col + ord("A")) + str(BOARD_SIZE - end_row)

    return f"{start_algebraic} {end_algebraic}"


def winner(game_state: GameState) -> Optional[str]:
    flat_board = [piece for row in game_state["board"] for piece in row]
    if "bK" not in flat_board:
        return "white"
    if "wK" not in flat_board:
        return "black"
    return None
