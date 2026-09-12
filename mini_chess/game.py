"""MiniChess game loop and trace output."""

from typing import Optional

from .ai import MiniChessAI, format_number
from .rules import (
    GameState,
    Move,
    init_board,
    is_valid_move,
    make_move,
    move_to_algebraic,
    parse_input,
    render_board,
    valid_moves,
    winner,
)


class MiniChess:
    def __init__(self, timeout: float, max_turns: int, use_alpha_beta: bool, play_mode: str, heuristic: str):
        self.current_game_state = self.init_board()
        self.no_capture_turns = 0
        self.turnNumber = 0
        self.timeout = timeout
        self.max_turns = max_turns
        self.use_alpha_beta = use_alpha_beta
        self.play_mode = play_mode
        self.heuristic = heuristic
        self.ai = MiniChessAI(timeout, use_alpha_beta, heuristic)

        alpha_beta_str = "true" if use_alpha_beta else "false"
        output_filename = f"gameTrace-{alpha_beta_str}-{timeout}-{max_turns}.txt"
        self.output_file = open(output_filename, "w", encoding="utf-8")
        self.write_game_parameters()

    def __del__(self):
        self.close()

    def close(self):
        output_file = getattr(self, "output_file", None)
        if output_file and not output_file.closed:
            output_file.close()

    def write_to_file(self, text: str):
        print(text)
        self.output_file.write(text + "\n")

    def write_game_parameters(self):
        self.write_to_file("===== GAME PARAMETERS =====")
        self.write_to_file(f"Timeout: {self.timeout} seconds")
        self.write_to_file(f"Max turns: {self.max_turns}")
        self.write_to_file(f"Play mode: {self.play_mode}")

        if "AI" in self.play_mode:
            self.write_to_file(f"Alpha-beta: {'ON' if self.use_alpha_beta else 'OFF'}")
            self.write_to_file(f"Heuristic: {self.heuristic}")

        self.write_to_file("===========================")

    def init_board(self) -> GameState:
        return init_board()

    def display_board(self, game_state: GameState):
        self.write_to_file(render_board(game_state))

    def is_valid_move(self, game_state: GameState, move: Move) -> bool:
        return is_valid_move(game_state, move)

    def valid_moves(self, game_state: GameState):
        return valid_moves(game_state)

    def make_move(self, game_state: GameState, move: Move) -> GameState:
        return make_move(game_state, move)

    def parse_input(self, move: str) -> Optional[Move]:
        return parse_input(move)

    def move_to_algebraic(self, move: Move) -> str:
        return move_to_algebraic(move)

    def evaluate(self, game_state: GameState, heuristic: str = "e0") -> float:
        return self.ai.evaluate(game_state, heuristic)

    def minimax(
        self,
        game_state: GameState,
        depth: int,
        is_maximizing: bool,
        alpha: float = float("-inf"),
        beta: float = float("inf"),
        start_time: Optional[float] = None,
        time_limit: Optional[float] = None,
    ):
        return self.ai.minimax(game_state, depth, is_maximizing, alpha, beta, start_time, time_limit)

    def get_ai_move(self, game_state: GameState):
        result = self.ai.get_move(game_state)
        return (
            result.move,
            result.search_score,
            result.heuristic_score,
            result.time_taken,
            result.states_this_move,
            result.max_depth,
        )

    def format_number(self, num: int) -> str:
        return format_number(num)

    def display_ai_stats(self, states_this_move=0, max_depth=0):
        for line in self.ai.statistics_lines():
            self.write_to_file(line)

    def play(self):
        self.write_to_file("Welcome to Mini Chess!")
        self.display_board(self.current_game_state)

        while True:
            game_winner = winner(self.current_game_state)
            if game_winner == "white":
                self.write_to_file(f"White wins after {self.turnNumber} turns!")
                break
            if game_winner == "black":
                self.write_to_file(f"Black wins after {self.turnNumber} turns!")
                break

            if self.no_capture_turns >= self.max_turns:
                self.write_to_file(f"Draw! No captures in the last {self.max_turns} turns.")
                break

            if self.turnNumber >= self.max_turns * 2:
                self.write_to_file(f"Draw! Maximum number of turns ({self.max_turns}) reached.")
                break

            current_player = self.current_game_state["turn"]
            self.write_to_file(f"Turn {self.turnNumber}.")
            self.write_to_file(f"{self.no_capture_turns} turns since last capture.")

            is_ai = self._is_ai_turn(current_player)

            if is_ai:
                self.write_to_file(f"{current_player.capitalize()} (AI) is thinking...")
                move, search_score, heuristic_score, time_taken, states_this_move, max_depth = self.get_ai_move(
                    self.current_game_state
                )

                if move:
                    move_str = self.move_to_algebraic(move)
                    self.write_to_file(f"{current_player.capitalize()} (AI) move: {move_str}")
                    self.write_to_file(f"Time for this action: {time_taken:.2f} sec")
                    self.write_to_file(f"Heuristic score: {heuristic_score}")
                    self.write_to_file(
                        f"{'Alpha-beta' if self.use_alpha_beta else 'Minimax'} search score: {search_score}"
                    )
                    self.display_ai_stats(states_this_move, max_depth)
                else:
                    self.write_to_file(f"{current_player.capitalize()} (AI) could not find a valid move!")
                    if current_player == "white":
                        self.write_to_file("Black wins due to AI error!")
                    else:
                        self.write_to_file("White wins due to AI error!")
                    break
            else:
                move_str = input(f"{current_player.capitalize()} to move: ")
                self.write_to_file(f"{current_player.capitalize()} to move: {move_str}")

                if move_str.lower() == "exit":
                    self.write_to_file("Game exited.")
                    return

                move = self.parse_input(move_str)

            if not move or not self.is_valid_move(self.current_game_state, move):
                if is_ai:
                    invalid_move = self.move_to_algebraic(move) if move else "None"
                    self.write_to_file(f"Invalid move by {current_player} AI: {invalid_move}")
                    if current_player == "white":
                        self.write_to_file("Black wins due to invalid AI move!")
                    else:
                        self.write_to_file("White wins due to invalid AI move!")
                    break

                self.write_to_file("Invalid move. Try again.")
                continue

            _, dest = move
            dest_piece = self.current_game_state["board"][dest[0]][dest[1]]
            is_capture = dest_piece != "."

            self.make_move(self.current_game_state, move)

            self.no_capture_turns = 0 if is_capture else self.no_capture_turns + 1
            self.turnNumber += 1

            self.display_board(self.current_game_state)

    def _is_ai_turn(self, current_player: str) -> bool:
        return (current_player == "white" and self.play_mode in ["AI-H", "AI-AI"]) or (
            current_player == "black" and self.play_mode in ["H-AI", "AI-AI"]
        )
