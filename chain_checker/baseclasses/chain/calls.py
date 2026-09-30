from collections.abc import Sequence
from typing import Any, NamedTuple

USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


class Call(NamedTuple):
    prompt: list[Any]
    reply: str
    model_name: str
    token_usage: dict[str, int]


def render_conversation(calls: Sequence[Call]) -> str:
    return "".join(_render_call(number, len(calls), call) for number, call in enumerate(calls, 1))


def _render_call(number: int, total: int, call: Call) -> str:
    parts = [
        f"=== call {number}/{total} ===\n",
        f"--- model ---\n{call.model_name}\n\n",
    ]
    parts += [f"--- {message.type} ---\n{message.content}\n\n" for message in call.prompt]
    parts.append(f"--- raw reply ---\n{call.reply}\n\n")
    return "".join(parts)


def sum_token_usage(calls: Sequence[Call]) -> dict[str, int]:
    return {key: sum(call.token_usage.get(key, 0) for call in calls) for key in USAGE_KEYS}


def answering_models(calls: Sequence[Call]) -> list[str]:
    return sorted({call.model_name for call in calls if call.model_name})
