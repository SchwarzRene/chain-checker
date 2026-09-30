import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from chain_checker.baseclasses.chain.calls import Call
from chain_checker.utils.console import link_print, link_print_warning


class _TextPrompt:
    """Wraps a plain-text completion prompt in the same `.type`/`.content`
    shape as a langchain chat message, so `_render_call` can format both
    chat-model and plain-completion calls without a branch."""

    type = "prompt"

    def __init__(self, content: str) -> None:
        self.content = content


def _count(
    primary: dict[str, Any], primary_key: str, fallback: dict[str, Any], fallback_key: str
) -> int:
    # Different providers report usage under different key names and in
    # different places (message.usage_metadata vs. llm_output.token_usage);
    # try the modern key first and only fall back to the legacy one.
    value = primary.get(primary_key)
    if value is None:
        value = fallback.get(fallback_key)
    return value or 0


def _first_generation(response: Any) -> Any:
    generations = response.generations or []
    if not generations or not generations[0]:
        raise ValueError(
            "the model returned no generation, which usually means the reply "
            "was filtered or the provider dropped it"
        )
    return generations[0][0]


def _extract_usage(response: Any, generation: Any) -> dict[str, int]:
    message = getattr(generation, "message", None)
    usage_metadata = getattr(message, "usage_metadata", None) or {}
    token_usage = (response.llm_output or {}).get("token_usage") or {}

    prompt_tokens = _count(usage_metadata, "input_tokens", token_usage, "prompt_tokens")
    completion_tokens = _count(usage_metadata, "output_tokens", token_usage, "completion_tokens")
    total = usage_metadata.get("total_tokens") or token_usage.get("total_tokens")

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        # Some providers omit an explicit total; derive it rather than report 0.
        "total_tokens": total or prompt_tokens + completion_tokens,
    }


class ChainAnalyser(BaseCallbackHandler):
    """Langchain callback handler that records every LLM call made while a
    chain runs, as `Call` entries, so a run can be reported on afterwards
    (prompt, reply, model name, token usage) without instrumenting the
    chain itself."""

    def __init__(self) -> None:
        self.calls: list[Call] = []
        self._prompts: dict[UUID, list[Any]] = {}
        self._start_times: dict[UUID, float] = {}

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[Any]], *, run_id: UUID, **kwargs: Any
    ) -> None:
        # `messages` is a batch of message lists; chains here only ever invoke
        # one input at a time, so only the first (and only) batch entry matters.
        self._prompts[run_id] = messages[0] if messages else []
        self._start_times[run_id] = time.monotonic()
        link_print("(MODEL) -> sent a prompt to the LLM, waiting for a reply...")

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID, **kwargs: Any
    ) -> None:
        # Same single-invocation assumption as on_chat_model_start, but for
        # plain-completion models: only the first prompt in the batch is kept.
        self._prompts[run_id] = [_TextPrompt(text) for text in prompts[:1]]
        self._start_times[run_id] = time.monotonic()
        link_print("(MODEL) -> sent a prompt to the LLM, waiting for a reply...")

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        prompt = self._prompts.pop(run_id, [])
        elapsed = time.monotonic() - self._start_times.pop(run_id, time.monotonic())
        generation = _first_generation(response)
        link_print(f"(MODEL) <- reply received after {elapsed:.1f}s ({len(generation.text)} chars)")
        self.calls.append(
            Call(
                prompt=prompt,
                reply=generation.text,
                model_name=(response.llm_output or {}).get("model_name") or "",
                token_usage=_extract_usage(response, generation),
            )
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        # Drop the captured prompt rather than re-raise here: langchain already
        # propagates the error through its own callback/exception path.
        elapsed = time.monotonic() - self._start_times.pop(run_id, time.monotonic())
        link_print_warning(f"(MODEL) LLM call failed after {elapsed:.1f}s: {error}")
        self._prompts.pop(run_id, None)
