import argparse

from chain_checker.modifier.llm.litellm import (
    DEFAULT_LITELLM_APP_LABEL,
    DEFAULT_LITELLM_TIER,
)
from chain_checker.utils.cli_args import add_common_args
from chain_checker.utils.config import apply_config_file, parse_with_sources

# Every dest parse_args_with_sources() reports a source for - kept in one
# place so a new --flag added to any of the _add_*_args groups below is a
# one-line addition here, not a silent gap in the startup report.
_TRACKED_PARAMS = (
    "type",
    "chain_type",
    "chain_tier",
    "file",
    "val_file",
    "prompt_file",
    "epochs",
    "modifier_backend",
    "modifier_model",
    "modifier_app",
    "modifier_tier",
    "continue_run",
)


_CHAIN_TYPE_HELP = (
    "Which chain type to run - must be a leaf chain with its own SYSTEM_PROMPT (see README.md)"
)


def _add_corpus_args(parser: argparse.ArgumentParser) -> None:
    add_common_args(parser, chain_type_help=_CHAIN_TYPE_HELP)
    parser.add_argument(
        "--prompt-file",
        dest="prompt_file",
        default=None,
        help="Path to a text file holding a candidate SYSTEM_PROMPT to start "
        "training from instead of the chain's real one - lets a "
        "hand-written prompt be tried through the whole training loop "
        "without editing the chain's source. Ignored when --continue "
        "resumes a run past epoch 0 - the resumed run's own saved "
        "prompt wins there, same as any other --continue resume.",
    )
    parser.add_argument(
        "--val-file",
        default=None,
        help="Path to a second, held-out corpus .yaml - every epoch tests the "
        "current prompt on it too, right after --file, purely for "
        "reporting (never seen by the modifier, which only ever reads "
        "--file's results/false examples - see the module docstring). "
        "Off by default: with no --val-file (and no 'val_file' config "
        "key), nothing changes - a single --file, found or given the "
        "usual way, is still just the training corpus, exactly as "
        "before this existed.",
    )
    parser.add_argument("--epochs", type=int, default=4)


def _add_modifier_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--modifier-backend",
        choices=["litellm", "ollama"],
        default="litellm",
        help="Which LLM backend rewrites the prompt each epoch (the 'modifier "
        "LLM'). 'litellm' (default) calls this project's own LiteLLM "
        "proxy, borrowing --modifier-app's LiteLLM credential - no "
        "separate API key needed for the modifier itself. 'ollama' calls "
        "a local Ollama server instead - needs `ollama serve` running.",
    )
    parser.add_argument(
        "--modifier-model",
        default="qwen3.5:4b",
        help="Ollama model name, only used when --modifier-backend=ollama - "
        "any model name `ollama` already has pulled.",
    )
    parser.add_argument(
        "--modifier-app",
        default=DEFAULT_LITELLM_APP_LABEL,
        help="Django app label whose LiteLLM credential the modifier borrows, "
        "only used when --modifier-backend=litellm - needs "
        "LITELLM_API_KEY_<LABEL> set (e.g. in .env.local).",
    )
    parser.add_argument(
        "--modifier-tier",
        default=DEFAULT_LITELLM_TIER,
        help="LiteLLM tier/model alias the modifier calls, only used when "
        "--modifier-backend=litellm. Not restricted to a fixed list here - "
        "whatever aliases the proxy config defines for --modifier-app's key "
        "(e.g. 'fast', 'balanced', 'thinking', or a custom one like "
        "'qwen3.5:2b') are all valid. Checked against the proxy's actual "
        "model list at startup, which fails clearly if the tier isn't one "
        "this app's key can call.",
    )


def _add_resume_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Continue the newest existing run_N that was trained against "
        "this same chain config (type/chain-type/tier), adding more epochs "
        "to it instead of starting a new one. A run made with another tier "
        "is never continued into - that would mix two tiers into one run - "
        "so a fresh run_N is started beside it instead. Without this flag, "
        "every invocation starts a fresh run_N, leaving past attempts on "
        "disk untouched, so different attempts stay separately comparable.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Iteratively rewrites a leaf chain's system prompt against its corpus, "
        "using an LLM (this project's own LiteLLM proxy by default, or a local Ollama model)."
    )
    _add_corpus_args(parser)
    _add_modifier_backend_args(parser)
    _add_resume_args(parser)
    return parser


def parse_args() -> argparse.Namespace:
    parser = _build_parser()
    apply_config_file(parser)

    return parser.parse_args()


def parse_args_with_sources() -> tuple[argparse.Namespace, dict[str, str], str | None]:
    """Same CLI as parse_args(), plus - for every dest in _TRACKED_PARAMS -
    whether its final value came from an explicit flag, a config file, or
    the hardcoded default, and which config file (if any) was used. Lets
    trainingLoop.py print every value it is running with even when the user
    passed no flags at all."""
    return parse_with_sources(_build_parser(), _TRACKED_PARAMS)
