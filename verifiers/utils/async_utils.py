import inspect
from typing import Callable

import asyncio


async def maybe_await(func: Callable, *args, **kwargs):
    result = func(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def gather_with_running_avg(tasks: list, total: int, desc: str) -> list:
    """
    Gather async tasks with tqdm progress bar showing running averages.
    """
    from tqdm.asyncio import tqdm as async_tqdm

    results = [None] * total
    pbar = async_tqdm(total=total, desc=desc)

    count = 0
    sum_reward = 0.0
    sum_tokens = 0

    async def track(idx: int, task):
        nonlocal count, sum_reward, sum_tokens
        results[idx] = result = await task
        count += 1

        # print(result)

        if result is None:
            return

        # (reward, tokens) tuple from interleaved run_one()
        if isinstance(result[0], (int, float)):
            sum_reward += result[0]

        # process state
        state = result[1]
        token_count = 0
        if "responses" in state:
            for response in state["responses"]:
                if hasattr(response, "usage") and response.usage:
                    token_count += response.usage.completion_tokens or 0
        sum_tokens += token_count

        parts = [desc]
        if sum_reward:
            parts.append(f"avg_reward={sum_reward / count:.3f}")
        if sum_tokens:
            parts.append(f"completions_mean_length={sum_tokens / count:.0f}")

        if len(parts) > 1:
            pbar.set_description(" | ".join(parts))
        pbar.update(1)

    await asyncio.gather(*[track(i, t) for i, t in enumerate(tasks)])
    pbar.close()
    return results
