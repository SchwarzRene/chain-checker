"""The CLI flags checker.py and trainingLoop.py both define.

Kept apart from utils/config.py on purpose: that module resolves *values*
(which file wins, which key maps to which dest), this one declares the flags
themselves. Only what both tools spell identically lives here - a flag whose
meaning differs between them (--prompt-file, --continue) stays in its own
parser, where its help text can say what it actually does there.
"""

import argparse

_CONFIG_HELP = (
    "Path to a .yaml/.json file supplying any of these args by their flag's "
    "own spelling (e.g. 'chain-type: template_checklist', 'continue: true') "
    "instead of typing them every time - an explicit CLI flag still "
    "overrides the same key from the file. When omitted, "
    "chain_checker/ is searched for a config.json/.yaml and used "
    "the same way if one exists (see README.md's config-files section)."
)

_TYPE_HELP = "Which chain to run, e.g. 'tonality'"

_CHAIN_TIER_HELP = (
    "Override the chain-under-test's declared `tier` class attribute instead "
    "of calling the LLM tier it registers with - lets the same corpus be run "
    "against a different tier/model alias without editing the chain's "
    "source. Not --modifier-tier, which picks the tier for trainingLoop's "
    "separate modifier LLM; the two are independent. Not restricted to a "
    "fixed list here - whatever aliases the LiteLLM proxy defines for "
    "--type's key (e.g. 'fast', 'balanced', 'thinking', or a custom one) are "
    "all valid; an unknown tier fails clearly at the first LLM call instead."
)

_FILE_HELP = "Path to the corpus .yaml file, overrides the search in workflows/<type>"


def add_common_args(parser: argparse.ArgumentParser, *, chain_type_help: str) -> None:
    """Adds --config, --type, --chain-type, --chain-tier and --file.

    `chain_type_help` is passed in because it is the one flag of the five
    whose help genuinely differs: trainingLoop needs a leaf chain with a
    rewritable SYSTEM_PROMPT, the checker runs against any registered chain.
    Everything else here is identical for both, so it is written once.
    """
    parser.add_argument("--config", default=None, help=_CONFIG_HELP)
    parser.add_argument("--type", default="tonality", help=_TYPE_HELP)
    parser.add_argument("--chain-type", default="template_checklist", help=chain_type_help)
    parser.add_argument("--chain-tier", dest="chain_tier", default=None, help=_CHAIN_TIER_HELP)
    parser.add_argument("--file", default=None, help=_FILE_HELP)
