# tictactoe

### Overview
- **Environment ID**: `tictactoe`
- **Short description**: Multi-turn Tic-Tac-Toe where the LLM plays as X against a preference-based or optimal opponent
- **Tags**: games, train, eval, tictactoe

### Datasets
- **Primary dataset(s)**: Procedurally generated game episodes

### Task
- **Type**: multi-turn
- **Parser**: XMLParser (fields: `think`, `move`)
- **Rubric overview**: Win/draw/loss reward, format compliance, invalid move penalty

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval tictactoe
```

Configure model and sampling:

```bash
uv run vf-eval tictactoe   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7   -a '{"optimal_opponent_prob": 0.8}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_examples` | int | `100` | Number of game episodes to generate |
| `optimal_opponent_prob` | float | `0.5` | Probability opponent uses optimal minimax (vs fixed preference order). Fixed preference is used instead of random to ensure deterministic opponent responses within GRPO groups. |
| `use_think` | bool | `true` | Include `<think>` tags for reasoning before `<move>` |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `win_reward_func` | 1.0 for win, 0.5 for draw, 0.0 for loss/timeout |
| `format_reward_func` | Parser-driven format compliance (weight: 0.2) |
| `invalid_move_penalty_func` | -0.1 penalty per invalid move attempt |
