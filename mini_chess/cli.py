"""Command-line interface for MiniChess."""

import argparse
from typing import Optional, Sequence

from .game import MiniChess

PLAY_MODES = ["H-H", "H-AI", "AI-H", "AI-AI"]
HEURISTICS = ["e0", "e1", "e2"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_arguments(argv)
    timeout = args.timeout if args.timeout is not None else prompt_timeout()
    use_alpha_beta = args.alpha_beta if args.alpha_beta is not None else prompt_alpha_beta()
    play_mode = args.play_mode if args.play_mode is not None else prompt_play_mode()
    heuristic = args.heuristic if args.heuristic is not None else prompt_heuristic()

    game = MiniChess(timeout, args.max_turns, use_alpha_beta, play_mode, heuristic)
    try:
        game.play()
    finally:
        game.close()

    return 0


def parse_arguments(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="MiniChess Game with AI")
    parser.add_argument("-t", "--timeout", type=positive_float, help="Maximum AI thinking time per move in seconds")
    parser.add_argument(
        "-m",
        "--max-turns",
        "--max_turns",
        dest="max_turns",
        type=positive_int,
        default=20,
        help="Maximum no-capture turns before declaring a draw",
    )
    search_group = parser.add_mutually_exclusive_group()
    search_group.add_argument(
        "-a",
        "--alpha-beta",
        dest="alpha_beta",
        action="store_true",
        default=None,
        help="Use alpha-beta pruning",
    )
    search_group.add_argument(
        "--minimax",
        dest="alpha_beta",
        action="store_false",
        help="Use plain minimax search",
    )
    parser.add_argument(
        "-p",
        "--play-mode",
        type=lambda value: value.upper(),
        choices=PLAY_MODES,
        help="Play mode: H-H, H-AI, AI-H, or AI-AI",
    )
    parser.add_argument(
        "-e",
        "--heuristic",
        type=lambda value: value.lower(),
        choices=HEURISTICS,
        help="AI heuristic: e0, e1, or e2",
    )
    return parser.parse_args(argv)


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def prompt_timeout() -> float:
    timeout = -1.0
    while timeout <= 0:
        try:
            timeout = float(input("Enter the maximum time for AI to make a new move: "))
        except ValueError:
            print("Please enter a positive number.")
    return timeout


def prompt_alpha_beta() -> bool:
    choice = ""
    while choice not in ["yes", "no"]:
        choice = input("Type 'yes' for alpha beta or 'no' for mini-max: ").lower()
    return choice == "yes"


def prompt_play_mode() -> str:
    play_mode = ""
    while play_mode not in PLAY_MODES:
        play_mode = input("Select mode option: H-H, H-AI, AI-H, AI-AI: ").upper()
    return play_mode


def prompt_heuristic() -> str:
    heuristic = ""
    while heuristic not in HEURISTICS:
        heuristic = input("Select heuristic option: e0, e1, e2: ").lower()
    return heuristic
