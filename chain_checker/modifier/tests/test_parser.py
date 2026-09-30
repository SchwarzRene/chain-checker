import argparse

import pytest

from chain_checker.modifier import parser as parser_module
from chain_checker.modifier.parser import (
    _add_corpus_args,
    _add_modifier_backend_args,
    _add_resume_args,
    parse_args,
)
from chain_checker.utils import config as config_module


def _built(*adders) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    for adder in adders:
        adder(parser)
    return parser


def test_corpus_args_have_the_documented_defaults():
    args = _built(_add_corpus_args).parse_args([])

    assert args.type == "tonality"
    assert args.chain_type == "template_checklist"
    assert args.file is None
    assert args.prompt_file is None
    assert args.val_file is None
    assert args.epochs == 4


def test_modifier_backend_args_have_the_documented_defaults():
    args = _built(_add_modifier_backend_args).parse_args([])

    assert args.modifier_backend == "litellm"
    assert args.modifier_model == "qwen3.5:4b"
    assert args.modifier_tier == "fast"


def test_modifier_backend_is_restricted_to_the_two_real_backends():
    with pytest.raises(SystemExit):
        _built(_add_modifier_backend_args).parse_args(["--modifier-backend", "not-a-backend"])


def test_modifier_tier_accepts_any_alias_not_just_the_three_built_in_ones():
    # Proxy-configured, not fixed - a custom alias like a directly-named
    # Ollama-via-LiteLLM model must parse; availability is checked later,
    # against the proxy itself (see LiteLLM._check_tier_available).
    args = _built(_add_modifier_backend_args).parse_args(["--modifier-tier", "qwen3.5:2b"])
    assert args.modifier_tier == "qwen3.5:2b"


def test_continue_defaults_to_false_and_is_a_flag():
    args = _built(_add_resume_args).parse_args([])
    assert args.continue_run is False

    args = _built(_add_resume_args).parse_args(["--continue"])
    assert args.continue_run is True


# Config-file precedence itself is pinned once in utils/tests/test_config.py,
# against the apply_config_file()/parse_with_sources() both CLIs now share -
# it used to be tested here only, leaving the checker's own copy uncovered.


# --------------------------------------------------------------------------
# parse_args(): the whole CLI, isolated from this repo's own config.json
# --------------------------------------------------------------------------


def test_parse_args_wires_every_arg_group_together(monkeypatch):
    monkeypatch.setattr(config_module, "find_default_config_path", lambda: None)
    monkeypatch.setattr(
        "sys.argv", ["modifier", "--type", "tonality", "--epochs", "2", "--continue"]
    )

    args = parse_args()

    assert args.type == "tonality"
    assert args.epochs == 2
    assert args.continue_run is True
    assert args.modifier_backend == "litellm"


def test_parse_args_with_sources_reports_where_each_value_came_from(monkeypatch):
    monkeypatch.setattr(config_module, "find_default_config_path", lambda: None)
    monkeypatch.setattr("sys.argv", ["modifier", "--epochs", "2"])

    args, sources, config_path = parser_module.parse_args_with_sources()

    assert args.epochs == 2
    assert sources["epochs"] == "flag"
    assert sources["type"] == "default"
    assert config_path is None
