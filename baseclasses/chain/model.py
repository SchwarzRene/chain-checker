from __future__ import annotations

import asyncio
import importlib
import time
from typing import TYPE_CHECKING, Any, Protocol, cast

from chain_checker.baseclasses import corpus
from chain_checker.baseclasses.chain.analyser import ChainAnalyser
from chain_checker.baseclasses.chain.builder import build_runnable, load_chain
from chain_checker.baseclasses.chain.calls import (
    Call,
    answering_models,
    render_conversation,
    sum_token_usage,
)
from chain_checker.baseclasses.corpus.output import ModelOutput
from chain_checker.utils.console import link_print, link_print_warning
from chain_checker.utils.console.colors import IS_TTY
from chain_checker.utils.console.wait_bar import WaitBar
from chain_checker.utils.errors import fail
from chain_checker.utils.placeholders import (
    detect_placeholders,
    escape_stray_braces,
    find_near_miss_placeholders,
    unescape_braces,
)

if TYPE_CHECKING:
    from chain_checker.baseclasses.loop.progress import ProgressTracker


class _PromptModule(Protocol):
    SYSTEM_PROMPT: str


_HEARTBEAT_INTERVAL_SECONDS = 30.0

# A live-redrawn bar can tick far more often than a plain heartbeat print
# without spamming the log.
_BAR_TICK_INTERVAL_SECONDS = 1.0 if IS_TTY else _HEARTBEAT_INTERVAL_SECONDS


class ChainRunError(RuntimeError):
    """Safes what a model has already generated
    so when errors occur they are more traceable"""

    def __init__(self, message: str, calls: list[Call]) -> None:
        super().__init__(message)
        self.calls = calls


def _dump(value: Any) -> Any:
    # Recurses because a chain's output can be a pydantic model containing
    # further pydantic models nested in plain dicts/lists (e.g. a list of
    # sub-items); metrics need everything as plain data to compare against a
    # corpus label, not a mix of models and containers.
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {k: _dump(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_dump(v) for v in value]
    return value


class Model:
    """Wraps a registered chain into a callable that validates an `Input`
    against the chain's own InputSchema, runs it, and returns a
    `ModelOutput` carrying the predictions, a human-readable transcript, and
    token usage - captured via `ChainAnalyser` rather than by modifying the
    chain itself."""

    def __init__(self, type: str, chain_type: str, tier: str | None = None) -> None:
        self._type = type
        self._chain_type = chain_type
        self._tier_override = tier
        self._last_model_names: list[str] = []
        self._last_token_usage: dict[str, int] = {}
        self._wait_bar = WaitBar()
        self._wait_prefix: str | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None

        self.chain = load_chain(type, chain_type)
        self._log_prompt_module_debug()
        self.runnable = self._build_runnable()
        self._prompt_template_vars: tuple[str, ...] = self._detect_template_vars()

    def _log_prompt_module_debug(self) -> None:
        """Both `SYSTEM_PROMPT` and `generateSystemPrompt` are looked up by
        exact, hardcoded name elsewhere (`_optional_prompt_module`,
        `_render_system_prompt`) - printed here so a chain author sees
        immediately whether a typo or wrong scope (e.g. defined inside a
        class instead of at module level) is why a prompt isn't resolving
        or rewriting as expected."""
        module = importlib.import_module(self.chain.__module__)
        has_system_prompt = hasattr(module, "SYSTEM_PROMPT")
        has_generate = getattr(module, "generateSystemPrompt", None) is not None

        link_print()
        link_print(
            f"(MODEL) SYSTEM_PROMPT {'found' if has_system_prompt else 'NOT found'} "
            f"in chain '{self._chain_type}'s module ({module.__name__})"
        )
        link_print(
            f"(MODEL) generateSystemPrompt() {'found' if has_generate else 'NOT found'} "
            f"in chain '{self._chain_type}'s module ({module.__name__})"
        )
        link_print()

    def __call__(self, inp: corpus.Input) -> ModelOutput:
        payload = self._build_payload(inp)
        out, cap = self._run_chain(payload)

        self._last_token_usage = sum_token_usage(cap.calls)
        self._last_model_names = answering_models(cap.calls)

        return ModelOutput(
            render_conversation(cap.calls),
            self._predictions(out),
            self.get_last_token_usage(),
        )

    def call_with_progress(
        self, inp: corpus.Input, prefix: str, tracker: ProgressTracker
    ) -> tuple[ModelOutput | None, Exception | None, float]:
        """Like `__call__`, but drives `self._wait_bar` for the call's duration
        and returns (result, error, elapsed) instead of raising an ordinary
        `Exception`, so a caller can keep going over the rest of a case list."""
        self._wait_prefix = prefix
        start_time = time.monotonic()
        try:
            result = self(inp)
        except Exception as e:
            return None, e, self._close_wait_bar(prefix, tracker, start_time)
        return result, None, self._close_wait_bar(prefix, tracker, start_time)

    def _close_wait_bar(self, prefix: str, tracker: ProgressTracker, start_time: float) -> float:
        elapsed = time.monotonic() - start_time
        tracker.record(elapsed)
        self._wait_bar.finish(prefix, elapsed, tracker.get_summary(elapsed))
        self._wait_prefix = None
        return elapsed

    def _build_payload(self, inp: corpus.Input) -> dict[str, Any]:
        if not isinstance(inp, corpus.Input):
            fail(
                f"Model.__call__ expects an Input instance.\n"
                f"  Found: {type(inp).__name__}.\n"
                f"  Pass entry.get_input() (the Input object), not its raw dict."
            )

        data = inp.get()

        try:
            payload = self.chain.InputSchema(**data).model_dump()
        except Exception as e:
            raise ValueError(
                f"Input {data!r} does not match chain '{self._chain_type}'s "
                f"InputSchema: {e}. The corpus .yaml's input section must "
                f"match the chain's InputSchema fields exactly - same keys, "
                f"same shape (see documenation/CHAIN-REQUIREMENTS.md)."
            ) from e

        return payload

    def _predictions(self, out: Any) -> dict[str, Any]:
        predictions = _dump(out)
        if not isinstance(predictions, dict):
            raise ValueError(
                f"Chain '{self._chain_type}' returned {type(out).__name__} "
                f"({predictions!r}), which has no named fields to compare a "
                f"corpus label against. build() must return a Runnable whose "
                f"output is a pydantic model or a dict (see "
                f"documenation/CHAIN-REQUIREMENTS.md)."
            )
        return predictions

    def _run_chain(self, payload: dict[str, Any]) -> tuple[Any, ChainAnalyser]:
        cap = ChainAnalyser()
        try:
            out = self._invoke(payload, {"callbacks": [cap]})
        except Exception as e:
            raise ChainRunError(
                f"Chain '{self._chain_type}' raised while running: {e}", cap.calls
            ) from e
        return out, cap

    def _invoke(self, payload: dict[str, Any], config: dict[str, Any]) -> Any:
        if hasattr(self.runnable, "ainvoke"):
            self._ensure_no_running_loop()
            loop = self._get_event_loop()
            return loop.run_until_complete(
                self._await_with_heartbeat(self.runnable.ainvoke(payload, config=config))
            )

        if not hasattr(self.runnable, "invoke"):
            raise TypeError(
                f"chain '{self._chain_type}'s build() returned "
                f"{type(self.runnable).__name__}, which has neither ainvoke() "
                f"nor invoke() - build() must return a Runnable."
            )
        return self.runnable.invoke(payload, config=config)

    async def _await_with_heartbeat(self, coro: Any) -> Any:
        task = asyncio.ensure_future(coro)
        start = time.monotonic()
        if self._wait_prefix is not None:
            timeout = _BAR_TICK_INTERVAL_SECONDS
        else:
            timeout = _HEARTBEAT_INTERVAL_SECONDS
        while True:
            done, _ = await asyncio.wait({task}, timeout=timeout)
            if task in done:
                return task.result()
            elapsed = time.monotonic() - start
            if self._wait_prefix is not None:
                self._wait_bar.tick(self._wait_prefix, elapsed)
            else:
                link_print(
                    f"(MODEL) ... still waiting on chain '{self._chain_type}' "
                    f"({elapsed:.0f}s elapsed, no reply yet) - local models can "
                    f"take a long time, this isn't necessarily stuck"
                )

    def _get_event_loop(self) -> asyncio.AbstractEventLoop:
        # One shared loop, not a new one per call: httpx's connection pool is
        # bound to the loop it was created on and doesn't support a second one.
        if self._event_loop is None or self._event_loop.is_closed():
            self._event_loop = asyncio.new_event_loop()
        return self._event_loop

    def close(self) -> None:
        """Shuts down this Model's event loop, if `ainvoke()` ever created
        one - cancelling anything still pending first."""
        loop = self._event_loop
        if loop is None or loop.is_closed():
            return

        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        self._event_loop = None

    def _ensure_no_running_loop(self) -> None:
        # asyncio allows only one running loop per thread; run_until_complete()
        # would raise its own confusing RuntimeError trying to start a second
        # one, so this raises a clear one first.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError(
            f"Model must be called from sync code; it owns its own event "
            f"loop to drive chain '{self._chain_type}'s ainvoke()."
        )

    def get_type(self) -> str:
        return self._type

    def get_chain_type(self) -> str:
        return self._chain_type

    def get_tier(self) -> str:
        return self._effective_tier()

    def get_prompt_template_vars(self) -> tuple[str, ...]:
        return self._prompt_template_vars

    def get_config(self) -> dict[str, Any]:
        return {
            "type": self._type,
            "chain": self._chain_type,
            "tier": self._effective_tier(),
        }

    def get_run_info(self) -> dict[str, Any]:
        return {
            "chain": self._chain_type,
            "tier": self._effective_tier(),
            "model": ", ".join(self._last_model_names),
        }

    def _effective_tier(self) -> str:
        # The actual tier of the chain,
        # --chain-tier overwrites actual chain tier
        return self._tier_override or self.chain.tier

    def get_last_token_usage(self) -> dict[str, int]:
        return dict(self._last_token_usage)

    def get_system_prompt(self) -> str:
        """The prompt as the target model actually sees it - the same text
        a modifier reads and rewrites.

        The chain's own `generateSystemPrompt()`, where one exists, has
        already resolved every `{placeholder}` it fills, so that content is rewritable prose to the
        modifier, not an opaque token it could only copy through untouched
        """
        module = self._prompt_module()
        rendered = self._render_system_prompt(module)
        if rendered is not None:
            return rendered
        return unescape_braces(module.SYSTEM_PROMPT)

    def _render_system_prompt(self, module: _PromptModule) -> str | None:
        """Runs the chain's `generateSystemPrompt()` on an empty payload, or
        returns None if there's no such function, it raises, or it doesn't
        return a str - any of which falls back to the raw, unresolved
        SYSTEM_PROMPT."""
        generate = getattr(module, "generateSystemPrompt", None)
        if generate is None:
            return None

        # An empty payload, not a corpus entry's: resolving the prompt must
        # never depend on one entry's own data - that belongs in insertInput(),
        # which runs afterwards and stays invisible to the modifier.
        try:
            rendered = generate({})
        except Exception as e:  # noqa: BLE001 - deliberate, see below
            # Broad on purpose: a generateSystemPrompt() that needs real
            # payload data (or fails for any other reason) must not take the
            # run down - the unresolved template is still a usable prompt to
            # show and rewrite. A fatal error raised through fail() is a
            # CheckerError (a SystemExit), so it still propagates - see
            # utils/tests/test_errors.py.
            link_print_warning(
                f"(MODEL) WARNING: chain '{self._chain_type}'s "
                f"generateSystemPrompt() raised on an empty payload "
                f"({type(e).__name__}: {e}) - falling back to the unresolved "
                f"SYSTEM_PROMPT, so anything it would have filled in stays a "
                f"literal {{placeholder}} a modifier cannot reword."
            )
            return None

        if not isinstance(rendered, str):
            link_print_warning(
                f"(MODEL) WARNING: chain '{self._chain_type}'s "
                f"generateSystemPrompt() returned {type(rendered).__name__}, "
                f"not a str - falling back to the unresolved SYSTEM_PROMPT."
            )
            return None

        return rendered

    def set_new_system_prompt(self, prompt: str) -> None:
        """Stores a modifier's rewrite as the chain's new SYSTEM_PROMPT and
        rebuilds the runnable so it takes effect.

        Raises `ValueError` only for a mangled attempt at keeping a declared
        placeholder live (wrong spacing, wrong name); a clean drop of one is
        allowed through here and only surfaced as a warning elsewhere (see
        `ModifierModel._warn_if_placeholders_frozen`).
        """
        module = self._prompt_module()

        # A modifier-rewritten prompt is free-form text and may contain "{"/"}"
        # that were never meant as format placeholders; escape anything that
        # isn't one of the declared template vars so str.format() doesn't choke.
        escaped = escape_stray_braces(prompt, self._prompt_template_vars)

        missing = set(self._prompt_template_vars) - set(detect_placeholders(escaped))
        mangled = set(find_near_miss_placeholders(prompt, self._prompt_template_vars)) & missing
        if mangled:
            raise ValueError(
                f"Chain '{self._chain_type}'s rewritten prompt mangled "
                f"placeholder(s) {sorted(mangled)} - check spelling and "
                f"spacing, e.g. '{{{next(iter(mangled))}}}' with no inner "
                f"spaces."
            )

        module.SYSTEM_PROMPT = escaped

        # The runnable/prompt template was built from the module's SYSTEM_PROMPT
        # at construction time, so mutating the constant alone has no effect -
        # it must be rebuilt to pick up the new value.
        self.runnable = self._build_runnable()

    def _detect_template_vars(self) -> tuple[str, ...]:
        """Names still needing a real value on every call - the ones
        `generateSystemPrompt()` could not resolve from an empty payload
        alone.

        A var it does resolve, filled from static
        config) is gone from the text a modifier ever sees, so there's
        nothing left to protect. Anything still present afterwards needed
        real per-entry data and must never silently drop out of a rewrite.
        """
        # Chains without a SYSTEM_PROMPT (nothing for the modifier to rewrite)
        # simply have no template vars to preserve.
        module = self._optional_prompt_module()
        if module is None:
            return ()

        try:
            declared = detect_placeholders(module.SYSTEM_PROMPT)
        except ValueError as e:
            fail(
                f"Chain '{self._chain_type}'s SYSTEM_PROMPT is not a valid format template.\n"
                f"  Reason: {e}\n"
                f"  Double any brace it should keep as a literal (see "
                f"documenation/CHAIN-REQUIREMENTS.md)."
            )

        rendered = self._render_system_prompt(module)
        if rendered is None:
            return declared

        # Checked as a literal "{name}" substring rather than re-running
        # detect_placeholders() on the rendered text: a resolved
        # {{"ok": true}}-style JSON example is single-braced by then and
        # would otherwise be misread as a fresh field reference.
        return tuple(name for name in declared if f"{{{name}}}" in rendered)

    def _optional_prompt_module(self) -> _PromptModule | None:
        """Not every chain exposes a rewritable prompt, so a missing
        SYSTEM_PROMPT is a valid state here - see `_prompt_module` for the
        variant that treats it as an error instead."""
        module = importlib.import_module(self.chain.__module__)
        if not hasattr(module, "SYSTEM_PROMPT"):
            return None
        return cast(_PromptModule, module)

    def _prompt_module(self) -> _PromptModule:
        module = self._optional_prompt_module()
        if module is None:
            raise AttributeError(
                f"Chain '{self._chain_type}' has no module-level SYSTEM_PROMPT constant.\n"
                f"  Module: {self.chain.__module__}\n"
                f"  See documenation/CHAIN-REQUIREMENTS.md."
            )
        return module

    def _build_runnable(self) -> Any:
        return build_runnable(self.chain, self._type, self._chain_type, tier=self._tier_override)
