import argparse

from chain_checker.utils.cli_args import add_common_args
from chain_checker.utils.config import parse_with_sources

_TRACKED_PARAMS = ("type", "chain_type", "file", "prompt_file", "chain_tier", "continue_run")

_CHAIN_TYPE_HELP = "Which chain type to run, e.g. 'template_checklist'"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Runs a corpus through a registered chain once and reports how many "
        "cases the chain got right."
    )
    add_common_args(parser, chain_type_help=_CHAIN_TYPE_HELP)
    parser.add_argument(
        "--prompt-file",
        dest="prompt_file",
        default=None,
        help="Path to a text file holding a candidate SYSTEM_PROMPT to test "
        "instead of the chain's real one - lets a hand-written prompt be "
        "tried against the corpus without editing the chain's source.",
    )
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Recall cases already predicted in the newest existing check_N "
        "dir made with this run's own prompt and chain config - i.e. the "
        "same type/chain-type/tier and the same SYSTEM_PROMPT (e.g. "
        "entries dropped in there by a run made elsewhere) - instead of "
        "calling the model for them again, then run only what's still "
        "missing and write the report. A check_N made with anything else "
        "is never continued into: it keeps its predictions and its report, "
        "and this run starts a fresh check_N beside it. Without this flag, "
        "every invocation starts a fresh check_N and calls the model for "
        "every case.",
    )
    return parser


def parse_args() -> tuple[argparse.Namespace, dict[str, str], str | None]:
    return parse_with_sources(_build_parser(), _TRACKED_PARAMS)
