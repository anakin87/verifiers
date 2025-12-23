

uv run vf-eval tictactoe -n 100 -r 1 -m Qwen3-0.6B -e "endpoints.py"  -a '{"optimal_opponent_prob": 0}' --save-results

uv run vf-eval tictactoe -n 100 -r 1 -m Qwen3-0.6B -e "endpoints.py"  -a '{"optimal_opponent_prob": 1}' --save-results

uv run vf-eval tictactoe -n 100 -r 1 -m qwen3-ttt-better-merged2 -e "endpoints.py"  -a '{"optimal_opponent_prob": 0}' --save-results

uv run vf-eval tictactoe -n 100 -r 1 -m qwen3-ttt-better-merged2 -e "endpoints.py"  -a '{"optimal_opponent_prob": 1}' --save-results

uv run vf-rl @ config.toml

-
setup wandb
pip install wandb
wandb login

setup hf
curl -LsSf https://hf.co/cli/install.sh | bash
git config --global credential.helper store
hf auth login

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env 
uv tool install prime
prime env pull(/install) anakin87/tictactoe

uv init
uv add 'verifiers[rl]'
uv pip install "vllm>=0.10.0,<0.11.0"

apt install tmux


uv run vf-rl @ config.toml

 uv run vf-vllm --model anakin87/qwen3-ttt-better-merged2

