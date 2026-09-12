"""Heuristics and minimax search for MiniChess."""

import copy
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .rules import GameState, Move, make_move, valid_moves

PIECE_VALUES = {
    "p": 1,
    "B": 3,
    "N": 3,
    "Q": 9,
    "K": 999,
}


@dataclass
class AiMoveResult:
    move: Optional[Move]
    search_score: float
    heuristic_score: float
    time_taken: float
    states_this_move: int
    max_depth: int


class MiniChessAI:
    def __init__(self, timeout: float, use_alpha_beta: bool, heuristic: str):
        self.timeout = timeout
        self.use_alpha_beta = use_alpha_beta
        self.heuristic = heuristic
        self.states_explored = 0
        self.states_by_depth: Dict[int, int] = {}
        self.total_branching = 0
        self.total_decisions = 0

    def evaluate(self, game_state: GameState, heuristic: Optional[str] = None) -> float:
        heuristic = heuristic or self.heuristic
        board = game_state["board"]
        piece_counts = {
            "wp": 0,
            "wB": 0,
            "wN": 0,
            "wQ": 0,
            "wK": 0,
            "bp": 0,
            "bB": 0,
            "bN": 0,
            "bQ": 0,
            "bK": 0,
        }

        for row in board:
            for piece in row:
                if piece != ".":
                    piece_counts[piece] += 1

        white_score = self._material_score(piece_counts, "w")
        black_score = self._material_score(piece_counts, "b")

        if heuristic == "e0":
            return white_score - black_score

        if heuristic == "e1":
            for row in range(5):
                for col in range(5):
                    piece = board[row][col]
                    if piece == ".":
                        continue

                    if piece in ["wN", "wB"] and 1 <= row <= 3 and 1 <= col <= 3:
                        white_score += 0.5
                    elif piece in ["bN", "bB"] and 1 <= row <= 3 and 1 <= col <= 3:
                        black_score += 0.5

                    if piece == "wp":
                        white_score += 0.1 * (4 - row)
                    elif piece == "bp":
                        black_score += 0.1 * row

            return white_score - black_score

        if heuristic == "e2":
            game_state_copy = copy.deepcopy(game_state)
            game_state_copy["turn"] = "white"
            white_moves = len(valid_moves(game_state_copy))

            game_state_copy["turn"] = "black"
            black_moves = len(valid_moves(game_state_copy))

            white_score += 0.1 * white_moves
            black_score += 0.1 * black_moves

            return white_score - black_score

        return self.evaluate(game_state, "e0")

    def minimax(
        self,
        game_state: GameState,
        depth: int,
        is_maximizing: bool,
        alpha: float = float("-inf"),
        beta: float = float("inf"),
        start_time: Optional[float] = None,
        time_limit: Optional[float] = None,
    ) -> Tuple[float, Optional[Move]]:
        self.states_explored += 1
        self.states_by_depth[depth] = self.states_by_depth.get(depth, 0) + 1

        if start_time and time_limit and time.time() - start_time > time_limit:
            return 0, None

        flat_board = [piece for row in game_state["board"] for piece in row]
        if "bK" not in flat_board:
            return float("inf"), None
        if "wK" not in flat_board:
            return float("-inf"), None

        if depth == 0:
            return self.evaluate(game_state, self.heuristic), None

        moves = valid_moves(game_state)

        if moves:
            self.total_branching += len(moves)
            if depth == 3:
                self.total_decisions += 1

        if not moves:
            return self.evaluate(game_state, self.heuristic), None

        best_move = None

        if is_maximizing:
            best_score = float("-inf")
            for move in moves:
                new_state = copy.deepcopy(game_state)
                make_move(new_state, move)

                score, _ = self.minimax(new_state, depth - 1, False, alpha, beta, start_time, time_limit)

                if score > best_score:
                    best_score = score
                    best_move = move

                if self.use_alpha_beta:
                    alpha = max(alpha, best_score)
                    if beta <= alpha:
                        break
        else:
            best_score = float("inf")
            for move in moves:
                new_state = copy.deepcopy(game_state)
                make_move(new_state, move)

                score, _ = self.minimax(new_state, depth - 1, True, alpha, beta, start_time, time_limit)

                if score < best_score:
                    best_score = score
                    best_move = move

                if self.use_alpha_beta:
                    beta = min(beta, best_score)
                    if beta <= alpha:
                        break

        return best_score, best_move

    def get_move(self, game_state: GameState) -> AiMoveResult:
        start_time = time.time()
        best_move = None
        best_score = float("-inf") if game_state["turn"] == "white" else float("inf")
        current_depth = 1
        max_depth = 10
        previous_states_explored = self.states_explored

        while current_depth <= max_depth:
            if time.time() - start_time > self.timeout * 0.8:
                break

            is_maximizing = game_state["turn"] == "white"
            score, move = self.minimax(
                game_state,
                current_depth,
                is_maximizing,
                float("-inf"),
                float("inf"),
                start_time,
                self.timeout * 0.95,
            )

            if move is not None:
                best_move = move
                best_score = score

            if (is_maximizing and score == float("inf")) or (not is_maximizing and score == float("-inf")):
                break

            current_depth += 1

        time_taken = time.time() - start_time

        if best_move:
            new_state = copy.deepcopy(game_state)
            make_move(new_state, best_move)
            heuristic_score = self.evaluate(new_state, self.heuristic)
        else:
            moves = valid_moves(game_state)
            if moves:
                best_move = moves[0]
                new_state = copy.deepcopy(game_state)
                make_move(new_state, best_move)
                heuristic_score = self.evaluate(new_state, self.heuristic)
                best_score = heuristic_score
            else:
                heuristic_score = self.evaluate(game_state, self.heuristic)
                best_score = heuristic_score

        states_this_move = self.states_explored - previous_states_explored

        return AiMoveResult(best_move, best_score, heuristic_score, time_taken, states_this_move, current_depth - 1)

    def statistics_lines(self):
        lines = ["\n===== AI STATISTICS ====="]
        lines.append(f"Cumulative states explored: {format_number(self.states_explored)}")

        depth_stats = "Cumulative states explored by depth: "
        for depth in sorted(self.states_by_depth.keys()):
            depth_stats += f"{depth}={format_number(self.states_by_depth[depth])} "
        lines.append(depth_stats)

        if self.states_explored > 0:
            percent_stats = "Cumulative % states explored by depth: "
            for depth in sorted(self.states_by_depth.keys()):
                percentage = (self.states_by_depth[depth] / self.states_explored) * 100
                percent_stats += f"{depth}={percentage:.1f}% "
            lines.append(percent_stats)

        if self.total_decisions > 0:
            avg_branching = self.total_branching / self.total_decisions
            lines.append(f"Average branching factor: {avg_branching:.1f}")

        lines.append("==========================\n")
        return lines

    @staticmethod
    def _material_score(piece_counts: Dict[str, int], color: str) -> int:
        return (
            PIECE_VALUES["p"] * piece_counts[f"{color}p"]
            + PIECE_VALUES["B"] * piece_counts[f"{color}B"]
            + PIECE_VALUES["N"] * piece_counts[f"{color}N"]
            + PIECE_VALUES["Q"] * piece_counts[f"{color}Q"]
            + PIECE_VALUES["K"] * piece_counts[f"{color}K"]
        )


def format_number(num: int) -> str:
    if num < 1000:
        return str(num)
    if num < 1000000:
        return f"{num / 1000:.1f}k"
    return f"{num / 1000000:.1f}M"
