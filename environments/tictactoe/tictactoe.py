import random
from typing import Any

import verifiers as vf
from datasets import Dataset
from verifiers.types import Messages, State


# GAME LOGIC

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


def check_win(board: list[str | None], player: str) -> bool:
    """Check if the given player has won."""
    return any(all(board[i] == player for i in line) for line in WINNING_LINES)


def get_free_positions(board: list[str | None]) -> list[int]:
    """Get list of empty positions on the board."""
    return [i for i in range(9) if board[i] is None]


def get_optimal_move(board: list[str | None]) -> int:
    """Find optimal move using minimax algorithm."""
    _, move = minimax(board=board, is_maximizing=True)
    assert move is not None  # Always valid when board has free cells
    return move


def minimax(board: list[str | None], is_maximizing: bool) -> tuple[float, int | None]:
    """Minimax algorithm. Returns (score, best_move) for the current player."""
    # Terminal states
    if check_win(board=board, player="O"):
        return 1, None
    if check_win(board=board, player="X"):
        return -1, None
    free = get_free_positions(board=board)
    if not free:
        return 0, None

    player = "O" if is_maximizing else "X"
    best_score = -float("inf") if is_maximizing else float("inf")
    best_move = free[0]

    for pos in free:
        board[pos] = player
        score, _ = minimax(board=board, is_maximizing=not is_maximizing)
        board[pos] = None

        if (
            is_maximizing
            and score > best_score
            or not is_maximizing
            and score < best_score
        ):
            best_score, best_move = score, pos

    return best_score, best_move


def get_preference_move(board: list[str | None], preference: list[int]) -> int:
    """Pick the first free cell according to a fixed preference order.

    Why not random? GRPO compares multiple completions for the same prompt (a "group").
    With a random opponent, the same model move could face different opponent responses, making reward comparisons
    noisy. With a fixed preference order, identical model moves always get identical opponent responses within a group,
    so reward differences reflect the model's choices, not randomness.
    """
    free = set(get_free_positions(board=board))
    for pos in preference:
        if pos in free:
            return pos
    raise ValueError("No free positions")


# VERIFIERS ENVIRONMENT

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
    free = get_free_positions(board=board)
    return f"{status}\n\n{render_board(board=board)}\n\nAvailable positions: {free}\n\nYour turn."


class TicTacToeEnv(vf.MultiTurnEnv):
    async def setup_state(self, state: State) -> State:
        info = state.get("info", {})
        state["board"] = list(
            info.get("initial_board")
        )  # copy the board to avoid mutations
        state["winner"] = None  # None=in progress, "X"/"O"=winner, "draw"=draw
        state["opponent_is_optimal"] = info.get("opponent_is_optimal", False)
        state["opponent_preference"] = info.get(
            "opponent_preference"
        )  # For non-optimal: deterministic preference order
        state["invalid_moves"] = 0
        state["next_response"] = None
        return state

    @vf.stop
    async def process_and_check_stop(self, state: State) -> bool:
        """Process the model's move and check if game should stop.

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
        if check_win(board=board, player="X"):
            state["winner"] = "X"
            return True
        if not get_free_positions(board=board):
            state["winner"] = "draw"
            return True

        # Opponent's move (O) and check for win
        opp_pos = (
            get_optimal_move(board=board)
            if state["opponent_is_optimal"]
            else get_preference_move(
                board=board, preference=state["opponent_preference"]
            )
        )
        board[opp_pos] = "O"
        if check_win(board=board, player="O"):
            state["winner"] = "O"
            return True
        if not get_free_positions(board=board):
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
        return 0.0  # Game didn't finish (timeout)
    return 1.0 if winner == "X" else 0.5 if winner == "draw" else 0.0


def invalid_move_penalty_func(state: State, **kwargs: Any) -> float:
    """Penalty for invalid moves: -0.1 per invalid move"""
    invalid_moves = state.get("invalid_moves", 0)
    return -0.1 * invalid_moves


def load_environment(
    num_examples: int = 100,
    optimal_opponent_prob: float = 0.5,
    use_think: bool = True,
    **kwargs,
) -> vf.Environment:
    # Create dataset - each example has fixed opponent type and preference order
    def make_dataset():
        rows = []
        for _ in range(num_examples):
            board = [None] * 9
            is_optimal = random.random() < optimal_opponent_prob
            # For non-optimal opponents, generate a fixed "preference order" instead of using random moves.
            # This is critical for GRPO: all completions within a group (same sample) must face
            # identical opponent behavior, so reward differences reflect model quality, not luck.
            # Different samples get different preference orders, preserving variety across the dataset.
            preference = None if is_optimal else random.sample(range(9), 9)

            if random.random() < 0.5:
                # Model starts
                question = user_feedback("Game started. You are X.", board)
            else:
                # Opponent starts - optimal plays center, non-optimal uses preference order
                opp_pos = 4 if is_optimal else preference[0]
                board[opp_pos] = "O"
                question = user_feedback(
                    f"Game started. Opponent (O) played at position {opp_pos}.", board
                )

            rows.append(
                {
                    "question": question,
                    "info": {
                        "initial_board": board,
                        "opponent_is_optimal": is_optimal,
                        "opponent_preference": preference,
                    },
                }
            )
        return Dataset.from_list(rows)

    parser = (
        vf.XMLParser(fields=["think", "move"], answer_field="move")
        if use_think
        else vf.XMLParser(fields=["move"], answer_field="move")
    )
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
