import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from chain_checker.baseclasses.chain.lenient_json import escape_stray_quotes

# Strips through the first </think> even with no opening tag: some models
# start straight into reasoning and only mark its end. Anchored to the
# start, so a later <think>...</think> pair is left for `_THINK_PAIR_RE`.
_THINK_LEADING_RE = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_PAIR_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"\A```[a-zA-Z]*\n?(.*?)\n?```\s*\Z", re.DOTALL)


def strip_reasoning(reply: str) -> str:
    """Strips a <think> block — even one missing its opening tag, see
    `_THINK_LEADING_RE` — and an optional wrapping code fence off a free-form
    reply."""
    cleaned = _THINK_LEADING_RE.sub("", reply, count=1)
    cleaned = _THINK_PAIR_RE.sub("", cleaned).strip()
    fenced = _CODE_FENCE_RE.match(cleaned)
    return fenced.group(1).strip() if fenced else cleaned


def parse_candidates(reply: str) -> list[str]:
    """Readings of `reply` to try as JSON, least invasive first: raw, then
    with a <think> block/code fence stripped (`strip_reasoning`), then
    either with stray quotes repaired (`escape_stray_quotes`). Duplicates
    are dropped, but raw stays first so the list is never empty."""
    raw = reply.strip()
    stripped = strip_reasoning(reply)
    candidates = [raw]
    for candidate in (stripped, escape_stray_quotes(raw), escape_stray_quotes(stripped)):
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


class StructuredJSONParseError(ValueError):
    """Raised when no reading of `reply` parses as `schema`'s JSON. Carries
    the cleaned text as a real attribute, not just in the message, so
    `checker.py` can log it alongside the raw reply, not only via this
    exception's `str()`."""

    def __init__(self, message: str, cleaned_reply: str) -> None:
        super().__init__(message)
        self.cleaned_reply = cleaned_reply


def parse_structured[S: BaseModel](schema: type[S], reply: str) -> S:
    """Parses `reply` as `schema` via `parse_candidates`, reporting the last
    failure if none parse. Public so a chain streaming its own reply gets
    the same handling as `with_structured_output()`; generic so it returns
    `schema`, not a bare BaseModel."""
    candidates = parse_candidates(reply)
    for candidate in candidates[:-1]:
        try:
            return schema.model_validate_json(candidate)
        except ValueError:
            continue
    try:
        return schema.model_validate_json(candidates[-1])
    except ValueError as e:
        # Only the summary line goes in the message; pydantic's per-field
        # breakdown stays reachable via `__cause__`.
        raise StructuredJSONParseError(
            f"Could not parse the model's reply as {schema.__name__} JSON, with or without "
            f"stripping a <think> block and repairing its quoting: {str(e).splitlines()[0]}",
            candidates[-1],
        ) from e


class _ThinkSafeMixin:
    """Overrides only `with_structured_output(schema, method="json_mode")`:
    some thinking models (e.g. qwen3.5:0.8b) never leave their `thinking`
    phase under json_mode's grammar-constrained decoding, leaving `content`
    empty. This calls the model in plain chat mode instead and recovers the
    schema's JSON from the reply (`parse_candidates`), also tolerating a
    wrapping code fence or unescaped quotes.

    Every other method falls through to `super()` untouched; mixed in ahead
    of the model's own class (see `make_think_safe`) purely to intercept
    this one method.
    """

    def with_structured_output(
        self, schema: type[BaseModel], *, method: str = "function_calling", **kwargs: Any
    ) -> Runnable:
        if method != "json_mode":
            # No real base until mixed onto a real BaseChatModel (see `make_think_safe`).
            return super().with_structured_output(schema, method=method, **kwargs)  # type: ignore[misc]

        async def _call(messages: Any) -> BaseModel:
            # Same: `self` only gets `ainvoke()` once mixed onto a real model.
            reply = await self.ainvoke(messages)  # type: ignore[attr-defined]
            return parse_structured(schema, reply.text)

        return RunnableLambda(_call)


# One subclass per original class, shared across repeated `make_think_safe()` calls.
_safe_classes: dict[type, type] = {}


def make_think_safe(llm: BaseChatModel) -> BaseChatModel:
    """Returns `llm` made safe for thinking models (see `_ThinkSafeMixin`),
    applied once here in `builder.py`'s `_build_llm` so every chain gets it
    with no chain-side change.

    Swaps `llm`'s class for a dynamic subclass rather than wrapping it, so
    `llm` stays isinstance-compatible, still pipes with `|` (LCEL), and
    keeps every other attribute untouched — a langchain chat model is a
    pydantic model, which rejects a shadowing instance attribute, ruling
    out a wrap-and-delegate approach.
    """
    original_cls = type(llm)
    safe_cls = _safe_classes.get(original_cls)
    if safe_cls is None:
        safe_cls = type(f"ThinkSafe{original_cls.__name__}", (_ThinkSafeMixin, original_cls), {})
        _safe_classes[original_cls] = safe_cls

    try:
        llm.__class__ = safe_cls
    except TypeError:
        # Immutable builtins reject a `__class__` swap; only a fake/test
        # double (with no real `with_structured_output()`) hits this.
        return llm
    return llm
