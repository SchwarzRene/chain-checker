from chain_checker.utils.errors import fail


class LLM:
    """Backend contract for the modifier loop: `__call__` sends a prompt to a
    model and returns its plain-text reply. The base implementation always
    refuses to run - a concrete backend (`Ollama`, `LiteLLM`) must override
    it, and `ModifierModel` must be pointed at one via `init_ollama()` or
    `init_litellm()` before its loop can call the modifier LLM for real."""

    def __init__(self) -> None:
        self._last_usage: dict[str, int] = {}

    def __call__(self, prompt: str) -> str:
        fail(
            "LLM is an abstract base class with no backing model.\n"
            "  Use a concrete implementation like Ollama(...) or "
            "litellm.LiteLLM(...), or call "
            "ModifierModel.init_ollama()/init_litellm() before running the "
            "modifier loop."
        )

    def get_last_usage(self) -> dict[str, int]:
        return dict(self._last_usage)

    def get_run_info(self) -> dict[str, str]:
        return {}

    def _record_usage(
        self, prompt_tokens: int, completion_tokens: int, total_tokens: int | None = None
    ) -> None:
        # Not every backend reports a total separately - derive it rather
        # than let a missing figure read as zero usage.
        derived_total = (
            total_tokens if total_tokens is not None else prompt_tokens + completion_tokens
        )
        self._last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": derived_total,
        }

    def _require_nonempty_text(self, text: str, message: str) -> str:
        # Shared so every backend fails the same way on an empty reply,
        # instead of each guarding its own copy of this check.
        if not text:
            fail(message)
        return text
