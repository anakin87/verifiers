LLM RL Environments 101: training a Small Language Model to play Tic Tac Toe

If you are interested in AI, you may have heard of Reinforcement Learning Environments for Language Models/Agents.
Yet, it's often unclear what an environment really means in the context of LLMs and how classic RL concepts translate to 
text generation models.

In this article, I'll gently introduce the idea of Environments for LMs. We'll soon put this in practice, by developing a Tic Tac Toe environment with the Verifiers library. The environment can be used to evaluate and train Language Models on this classic game.

Why Tic Tac Toe? It's a simple game with a small state space and a short determininistic algorithmic solution, but (small)
Language Models generally struggle with it. We'll see if Reinforcement Learning can help...

## Agents, Environments, and LLMs

In Reinforcement Learning, there are two main characters: the agent and the environment.
The environment is the world the agent interacts with.
At each step, the agent sees the current state of the world and takes an action. The state of the environment then changes based on the Agent's action. The agent also receives a reward from the environment: a number that tells it how good or bad the current world state is. The goal of the agent is to learn a policy that maximizes its cumulative reward over time.

[img by lil log...]

When it comes to Language Models, the standard training recipe includes three stages: pretraining on a large amount of internet text, Supervised Fine Tuning on conversational examples and Preference Alignment with techniques like PPO.

The release of DeepSeek-R1 brought renewed attention to RL in LLMs by applying GRPO and the idea of Reinforcement Learning with Verifiable Rewards. In this setup, the model is asked a question, generates a reasoning trace and an answer, and is then evaluated against a known ground truth. This is fundamentally different from SFT, where the model learns from examples.

With ideas in mind, we can map the classic RL framework onto LLMs. The Language Model plays the role of the Agent; the environment for a give task consists of data, harnesses and scoring rules: everything needed to evaluate and potentially train the model on that task.
As [Andrej Karpathy puts it](https://x.com/karpathy/status/1960803117689397543), environments
> give the LLM an opportunity to actually interact - take actions, see outcomes, etc. This means you can hope to do a lot better than statistical expert imitation.

The definition of an Agent is also expanding. LMs can now be given tools, from a weather API to a terminal. This makes environments for training and evaluation more complex and critical.

To make this more concrete, consider teaching a Vision Language Model to play the [2048 game](https://en.wikipedia.org/wiki/2048_(video_game)). In this example, the agent is the VLM, equipped with tools to "see" the screen (by taking screenshots) and act (by controlling the arrow keys).
The environment (the game itself) returns scores that can be associated with individual moves or with the final outcome of the game.
This setup allows the agent to learn effective strategies through trial and error, by trying to maximize its score, without needing pre-existing examples.

Some of these ideas may feel abstract at first, but they will become clearer once we walk through a concrete environment step by step. For more detailed information on RL concepts, check out the References section.

## Verifiers

Now that we understand how LLMs can act as agents within an environment and how rewards guide their learning, it's time to see this in practice.

To develop our Tic Tac Toe Environment, we'll use [Verifiers](https://github.com/PrimeIntellect-ai/verifiers), an open-source library maintained by William Brown and the Prime Intellect team.

Verifiers provides modular components for creating RL Environments for LLM agents. These Environments can be used for evaluating and training Language Models on a certain task.

*Why use a library at all? Why not just code everything from scratch?*
As always, the answer is a mix of simplicity and convenience.

Verifiers handles a lot of the boilerplate that comes up when working with RL environments for LLMs.

- Environments are Python packages that can be easily installed and distributed.

- The library provides base classes that you can extend to create single-turn environment with just one interaction between the model and the env, Multi Turn Environments, Tool Environments, where the model is equipped with tools, and several others.

- It includes abstractions for parsing model responses and defining reward functions.

- Verifiers abstracts model serving: it expects an OpenAI-compatible API endpoint, so you can use it with OpenAI models, models hosted on Open Router, or local models served with vLLM.

- The framework automatically handles interaction loops and trajectories, so you don't need to manage these details manually and just care about core environment logic.

- For training, Verifiers comes with its own trainer and integrates with other frameworks such as PRIME-RL, Tinker, and SkyRL.

Overall, Verifiers lets us focus on what the environment is testing and rewarding, rather than on infrastructure details.

If you want to explore the library further, check out the [documentation](https://docs.primeintellect.ai/verifiers/source/overview).

### Multi Turn Env





## References

RL: https://spinningup.openai.com/en/latest/spinningup/rl_intro.html; https://lilianweng.github.io/posts/2018-02-19-rl-overview/

RLVR: https://rlhfbook.com/c/14-reasoning

To understand more about GRPO and reasoning models, I recommend a series of articles by Sebastian Raschka: 
[1](https://sebastianraschka.com/blog/2025/understanding-reasoning-llms.html),
[2](https://sebastianraschka.com/blog/2025/first-look-at-reasoning-from-scratch.html), [3](https://sebastianraschka.com/blog/2025/the-state-of-reinforcement-learning-for-llm-reasoning.html).

