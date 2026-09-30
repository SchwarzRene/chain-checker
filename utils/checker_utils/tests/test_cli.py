import sys

from chain_checker.utils import config
from chain_checker.utils.checker_utils.cli import parse_args


def test_with_no_flags_and_no_config_file_everything_is_the_hardcoded_default(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(config, "_DEFAULT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["checker.py"])

    args, sources, config_path = parse_args()

    assert args.type == "tonality"
    assert args.chain_type == "template_checklist"
    assert args.file is None
    assert args.prompt_file is None
    assert args.chain_tier is None
    assert args.continue_run is False
    assert config_path is None
    assert sources == {
        "type": "default",
        "chain_type": "default",
        "file": "default",
        "prompt_file": "default",
        "chain_tier": "default",
        "continue_run": "default",
    }


def test_an_explicit_flag_is_reported_as_such(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_DEFAULT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["checker.py", "--type", "example_tonality"])

    args, sources, _ = parse_args()

    assert args.type == "example_tonality"
    assert sources["type"] == "flag"


def test_an_explicit_config_file_supplies_values_and_is_returned(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_DEFAULT_CONFIG_DIR", str(tmp_path))
    config_file = tmp_path / "my_config.yaml"
    config_file.write_text("chain-type: from_config\n")
    monkeypatch.setattr(sys, "argv", ["checker.py", "--config", str(config_file)])

    args, sources, config_path = parse_args()

    assert args.chain_type == "from_config"
    assert sources["chain_type"] == "config"
    assert config_path == str(config_file)


def test_a_config_file_found_by_the_default_search_is_used_without_the_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_DEFAULT_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.json").write_text('{"type": "from_default_search"}')
    monkeypatch.setattr(sys, "argv", ["checker.py"])

    args, sources, config_path = parse_args()

    assert args.type == "from_default_search"
    assert sources["type"] == "config"
    assert config_path == str(tmp_path / "config.json")


def test_an_explicit_flag_still_overrides_a_default_search_config(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_DEFAULT_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.json").write_text('{"type": "from_default_search"}')
    monkeypatch.setattr(sys, "argv", ["checker.py", "--type", "from_flag"])

    args, sources, _ = parse_args()

    assert args.type == "from_flag"
    assert sources["type"] == "flag"
