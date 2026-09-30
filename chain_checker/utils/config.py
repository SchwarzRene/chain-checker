import argparse
import glob
import os
from collections.abc import Iterable

import yaml

from chain_checker.utils.errors import fail

_DEFAULT_CONFIG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except OSError as e:
        fail(f"Could not open config file '{path}'.\n  Reason: {e.strerror or e}")

    except yaml.YAMLError as e:
        fail(f"Config file '{path}' is not valid YAML.\n  Reason: {e}")

    if data is None:
        return {}
    if not isinstance(data, dict):
        fail(
            f"Config file '{path}' has the wrong shape.\n"
            f"  Expected: a mapping of arg name to value (e.g. "
            f"'type: tonality').\n"
            f"  Found: {type(data).__name__}."
        )
    return data


def _flag_to_dest_map(parser: argparse.ArgumentParser) -> dict[str, str]:
    """Maps each flag's own long-option spelling (e.g. 'chain-type', or
    'continue' for --continue) to its real dest (e.g. 'chain_type',
    'continue_run')"""
    return {
        option_string[2:]: action.dest
        for action in parser._actions
        for option_string in action.option_strings
        if option_string.startswith("--")
    }


def apply_config_defaults(
    parser: argparse.ArgumentParser, config_path: str, strict: bool = True
) -> None:
    config = load_config(config_path)
    flag_to_dest = _flag_to_dest_map(parser)

    unknown = sorted(set(config) - set(flag_to_dest))
    if unknown:
        if not strict:
            config = {key: value for key, value in config.items() if key not in unknown}
        else:
            fail(
                f"Config file '{config_path}' has unknown key(s).\n"
                f"  Unknown: {unknown}\n"
                f"  Valid keys: {sorted(set(flag_to_dest) - {'help'})}"
            )
            return

    parser.set_defaults(**{flag_to_dest[key]: value for key, value in config.items()})


def find_default_config_path() -> str | None:
    matches = sorted(
        path
        for pattern in ("config.json", "config.yaml", "config.yml")
        for path in glob.glob(os.path.join(_DEFAULT_CONFIG_DIR, pattern))
    )
    return matches[0] if matches else None


def explicit_cli_dests(parser: argparse.ArgumentParser) -> set[str]:
    """Every dest the user actually typed a flag for, regardless of what
    value they gave it - found by temporarily swapping every default for a
    one-off sentinel and seeing which dests come back as something else.

    Comparing a dest's final value against its default can't tell "explicit
    --type example_tonality" apart from "config already defaulted it there"
    when the two happen to agree, so this checks independently of value,
    before config or the caller's own set_defaults() touch the parser.
    """
    sentinel = object()
    original_defaults = {action.dest: action.default for action in parser._actions}
    for action in parser._actions:
        action.default = sentinel
    try:
        probe, _ = parser.parse_known_args()
    finally:
        for action in parser._actions:
            action.default = original_defaults[action.dest]

    return {dest for dest, value in vars(probe).items() if value is not sentinel}


def describe_sources(
    tracked: Iterable[str],
    explicit: set[str],
    hardcoded_defaults: dict,
    config_defaults: dict,
) -> dict[str, str]:
    """For each dest in `tracked`, says whether its final value came from an
    explicit CLI flag, a config file, or the hardcoded default - so a run
    with no flags at all can still show where every value came from."""
    sources = {}
    for dest in tracked:
        if dest in explicit:
            sources[dest] = "flag"
        elif config_defaults.get(dest) != hardcoded_defaults.get(dest):
            sources[dest] = "config"
        else:
            sources[dest] = "default"
    return sources


def apply_config_file(parser: argparse.ArgumentParser) -> str | None:
    """Rewrites `parser`'s defaults from a config file and says which one was
    used, if any.

    An explicit --config is strict: a typo fails loudly and lists the valid
    keys. An auto-discovered one is not, since the same file is shared by
    both entry points and may hold keys only the other one defines.
    """
    known, _ = parser.parse_known_args()
    if known.config:
        apply_config_defaults(parser, known.config)
        return known.config

    config_path = find_default_config_path()
    if config_path:
        apply_config_defaults(parser, config_path, strict=False)
    return config_path


def parse_with_sources(
    parser: argparse.ArgumentParser, tracked: Iterable[str]
) -> tuple[argparse.Namespace, dict[str, str], str | None]:
    hardcoded_defaults = {action.dest: action.default for action in parser._actions}
    explicit = explicit_cli_dests(parser)

    config_path = apply_config_file(parser)
    config_defaults = {action.dest: action.default for action in parser._actions}

    args = parser.parse_args()
    sources = describe_sources(tracked, explicit, hardcoded_defaults, config_defaults)

    return args, sources, config_path
