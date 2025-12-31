import random
from functools import lru_cache
from typing import Any, Sequence

import verifiers as vf
from datasets import Dataset
from verifiers.types import Messages, State


# --- GAME LOGIC ---

WINNING_LINES = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],  # rows
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],  # columns
    [0, 4, 8],
    [2, 4, 6],  # diagonals
]


def render_board(board: list[str | None]) -> str:
    """Render the board as a string with position numbers for empty cells."""

    def cell(i: int) -> str:
        return board[i] or str(i)

    return (
        f"{cell(0)} | {cell(1)} | {cell(2)}\n"
        f"---------\n"
        f"{cell(3)} | {cell(4)} | {cell(5)}\n"
        f"---------\n"
        f"{cell(6)} | {cell(7)} | {cell(8)}"
    )


def check_win(board: Sequence[str | None], player: str) -> bool:
    """Check if the given player has won."""
    return any(all(board[i] == player for i in line) for line in WINNING_LINES)


def get_free_positions(board: Sequence[str | None]) -> list[int]:
    """Get list of empty positions on the board."""
    return [i for i in range(9) if board[i] is None]


# --- STATELESS SEEDING ---


def get_optimal_move(board: list[str | None], rng: random.Random) -> int:
    """Find optimal move, using the passed RNG for tie-breaking."""
    # Caller converts to tuple
    _, best_moves = minimax(tuple(board), is_maximizing=True)
    assert best_moves
    return rng.choice(best_moves)


@lru_cache(maxsize=None)
def minimax(
    board: tuple[str | None, ...], is_maximizing: bool
) -> tuple[float, list[int]]:
    """Pure minimax that returns score and all best moves, with memoization."""
    if check_win(board, "O"):
        return 1.0, []
    if check_win(board, "X"):
        return -1.0, []

    free = get_free_positions(board)
    if not free:
        return 0.0, []

    player = "O" if is_maximizing else "X"

    # Evaluate all possible moves
    results = []
    for pos in free:
        # Caller converts to tuple for recursion
        board_list = list(board)
        board_list[pos] = player
        score, _ = minimax(tuple(board_list), not is_maximizing)
        results.append((score, pos))

    # Identify best score and all moves that reach it
    scores = [res[0] for res in results]
    best_score = max(scores) if is_maximizing else min(scores)
    best_moves = [pos for score, pos in results if score == best_score]

    return best_score, best_moves


def get_random_move(board: list[str | None], rng: random.Random) -> int:
    """Pick a random valid move using the passed RNG."""
    free = get_free_positions(board)
    return rng.choice(free)


# --- VERIFIERS ENVIRONMENT ---

SYSTEM_PROMPT = f"""You are playing a game of Tic-Tac-Toe as X.
Your opponent will play as O.

Initial board:
{render_board([None] * 9)}

Your objective is to achieve three X in a row (horizontally, vertically, or diagonally) before the opponent does.
You can only choose one position each turn, and it must be an empty square.

You may include a short reasoning process inside <think>...</think> tags.
Your final answer must include the position you choose inside <move>...</move> tags.
"""


def user_feedback(status: str, board: list[str | None]) -> str:
    """Format consistent feedback to the model."""
    free = get_free_positions(board)
    return f"{status}\n\n{render_board(board)}\n\nAvailable positions: {free}\n\nYour turn."


class TicTacToeEnv(vf.MultiTurnEnv):
    async def setup_state(self, state: State) -> State:
        info = state.get("info", {})
        state["board"] = list(info.get("initial_board"))
        state["winner"] = None
        state["opponent_randomness"] = info.get("opponent_randomness", 0.0)
        state["example_seed"] = info.get("example_seed", 42)
        state["invalid_moves"] = 0
        state["next_response"] = None
        return state

    @vf.stop
    async def process_and_check_stop(self, state: State) -> bool:
        """
        Process the model's move and check if game should stop.

        This method both processes game logic and determines stopping.

        By processing here, we avoid generating a model response after game ends.
        """

        trajectory = state.get("trajectory", [])
        if not trajectory:
            return False  # No model response yet

        # Parse move from latest model response
        board = state["board"]
        move = self.parser.parse_answer(trajectory[-1]["completion"]) or ""
        free = get_free_positions(board)

        # Validate move: must be one of the free positions
        if move not in [str(p) for p in free]:
            state["invalid_moves"] += 1
            state["next_response"] = [
                {"role": "user", "content": user_feedback("Invalid move.", board)}
            ]
            return False

        pos = int(move)

        # Apply model's move (X) and check for win
        board[pos] = "X"
        if check_win(board, "X"):
            state["winner"] = "X"
            return True
        if not get_free_positions(board):
            state["winner"] = "draw"
            return True

        # Opponent's move (O)
        turn_seed = f"{state['example_seed']}_{state['board']}"
        rng = random.Random(turn_seed)

        if rng.random() < state["opponent_randomness"]:
            opp_pos = get_random_move(board, rng)
        else:
            opp_pos = get_optimal_move(board, rng)

        # Apply opponent's move (O) and check for win
        board[opp_pos] = "O"
        if check_win(board, "O"):
            state["winner"] = "O"
            return True
        if not get_free_positions(board):
            state["winner"] = "draw"
            return True

        # Game continues
        state["next_response"] = [
            {
                "role": "user",
                "content": user_feedback(
                    f"Opponent (O) played at position {opp_pos}.", board
                ),
            }
        ]
        return False

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        """Return the response prepared by process_and_check_stop."""
        return state["next_response"]


def win_reward_func(state: State, **kwargs: Any) -> float:
    """Reward function: 1.0 for win, 0.5 for draw, 0.0 for loss/timeout."""
    winner = state.get("winner")
    if winner is None:
        return 0.0
    return 1.0 if winner == "X" else 0.5 if winner == "draw" else 0.0


def invalid_move_penalty_func(state: State, **kwargs: Any) -> float:
    """Penalty function: -0.1 per invalid move attempt."""
    return -0.1 * state.get("invalid_moves", 0)


def load_environment(
    num_examples: int = 1000,
    min_opponent_randomness: float = 0.0,
    max_opponent_randomness: float = 0.4,
    use_think: bool = True,
    **kwargs,
) -> vf.Environment:
    def make_dataset():
        rows = []
        for _ in range(num_examples):
            board: list[str | None] = [None] * 9
            opponent_randomness = random.uniform(
                min_opponent_randomness, max_opponent_randomness
            )
            example_seed = random.randint(0, 1000000)

            rng = random.Random(f"{example_seed}_{board}")

            if rng.random() < 0.5:
                # Model starts
                question = user_feedback("Game started. You are X.", board)
            else:
                # Opponent starts
                if rng.random() < opponent_randomness:
                    opp_pos = get_random_move(board, rng)
                else:
                    opp_pos = get_optimal_move(board, rng)

                board[opp_pos] = "O"
                question = user_feedback(
                    f"Game started. Opponent (O) played at position {opp_pos}.", board
                )

            rows.append(
                {
                    "question": question,
                    "info": {
                        "initial_board": board,
                        "opponent_randomness": opponent_randomness,
                        "example_seed": example_seed,
                    },
                }
            )
        return Dataset.from_list(rows)

    # Handle thinking and non-thinking models
    move_parser = vf.XMLParser(fields=["move"], answer_field="move")

    def extract_move(text: str) -> str:
        return move_parser.parse_answer(text) or ""

    parser = vf.ThinkParser(extract_fn=extract_move) if use_think else move_parser

    rubric = vf.Rubric(parser=parser, funcs=[win_reward_func])

    rubric.add_reward_func(parser.get_format_reward_func(), weight=0.2)
    rubric.add_reward_func(invalid_move_penalty_func, weight=1.0)

    return TicTacToeEnv(
        dataset=make_dataset(),
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
        max_turns=10,
        **kwargs,
    )
