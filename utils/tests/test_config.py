import argparse
import json

import pytest

from chain_checker.utils import config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="tonality")
    parser.add_argument("--chain-type", dest="chain_type", default="template_checklist")
    return parser


def _cli_parser() -> argparse.ArgumentParser:
    """A parser shaped like both entry points': a --config flag, plus a value
    flag to watch the precedence rules move."""
    parser = _parser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs", type=int, default=4)
    return parser


# --------------------------------------------------------------------------
# load_config
# --------------------------------------------------------------------------


def test_a_yaml_file_loads_as_a_dict(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("type: example_tonality\nepochs: 4\n")

    assert config.load_config(str(path)) == {"type": "example_tonality", "epochs": 4}


def test_a_json_file_loads_too_since_json_is_valid_yaml(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"type": "example_tonality", "epochs": 4}')

    assert config.load_config(str(path)) == {"type": "example_tonality", "epochs": 4}


def test_an_empty_file_loads_as_an_empty_dict(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("")

    assert config.load_config(str(path)) == {}


def test_a_missing_file_fails_with_the_path_in_the_message(tmp_path, capsys):
    missing = tmp_path / "nope.yaml"

    with pytest.raises(SystemExit):
        config.load_config(str(missing))

    assert str(missing) in capsys.readouterr().err


def test_invalid_yaml_fails_clearly(tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text("type: [unterminated")

    with pytest.raises(SystemExit):
        config.load_config(str(path))

    assert "is not valid YAML" in capsys.readouterr().err


def test_a_non_mapping_top_level_value_is_rejected(tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text("- a\n- b\n")

    with pytest.raises(SystemExit):
        config.load_config(str(path))

    assert "list" in capsys.readouterr().err


# --------------------------------------------------------------------------
# apply_config_defaults
# --------------------------------------------------------------------------


def test_config_values_become_the_parsers_new_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("type: example_tonality\n")
    parser = _parser()

    config.apply_config_defaults(parser, str(path))

    assert parser.parse_args([]).type == "example_tonality"


def test_an_explicit_flag_still_overrides_the_config_value(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("type: example_tonality\n")
    parser = _parser()

    config.apply_config_defaults(parser, str(path))

    assert parser.parse_args(["--type", "floor_plan"]).type == "floor_plan"


def test_a_key_the_config_does_not_mention_keeps_its_original_default(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("type: example_tonality\n")
    parser = _parser()

    config.apply_config_defaults(parser, str(path))

    assert parser.parse_args([]).chain_type == "template_checklist"


def test_a_key_uses_the_flags_own_spelling_not_its_python_dest(tmp_path):
    # --chain-type's dest is chain_type, but the config key is the flag's own
    # spelling ('chain-type'), not the Python name.
    path = tmp_path / "config.yaml"
    path.write_text("chain-type: real_template_checklist\n")
    parser = _parser()

    config.apply_config_defaults(parser, str(path))

    assert parser.parse_args([]).chain_type == "real_template_checklist"


def test_the_python_dest_is_not_a_valid_config_key_when_it_differs_from_the_flag(tmp_path, capsys):
    # chain_type (the dest) must NOT work as a config key - only chain-type
    # (the flag's own spelling) does.
    path = tmp_path / "config.yaml"
    path.write_text("chain_type: real_template_checklist\n")
    parser = _parser()

    with pytest.raises(SystemExit):
        config.apply_config_defaults(parser, str(path))

    assert "chain_type" in capsys.readouterr().err


def test_a_flag_whose_dest_differs_from_its_own_name_uses_the_flags_spelling(tmp_path):
    # --continue has dest continue_run (continue is a reserved word) - the
    # config key is 'continue' (the flag's own spelling), not 'continue_run'.
    parser = argparse.ArgumentParser()
    parser.add_argument("--continue", dest="continue_run", action="store_true")

    path = tmp_path / "config.yaml"
    path.write_text("continue: true\n")
    config.apply_config_defaults(parser, str(path))
    assert parser.parse_args([]).continue_run is True


def test_an_unknown_key_fails_loudly_by_default(tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text("type: example_tonality\nnot_a_real_flag: 1\n")
    parser = _parser()

    with pytest.raises(SystemExit):
        config.apply_config_defaults(parser, str(path))

    err = capsys.readouterr().err
    assert "not_a_real_flag" in err
    assert "'type'" in err and "'chain-type'" in err


def test_the_valid_keys_listed_exclude_argparses_own_help_dest(tmp_path, capsys):
    # argparse gives every parser a `-h/--help` action whose dest is "help" -
    # not a config key a caller could ever set, so it must not show up in the
    # "valid keys are [...]" hint.
    path = tmp_path / "config.yaml"
    path.write_text("not_a_real_flag: 1\n")
    parser = _parser()

    with pytest.raises(SystemExit):
        config.apply_config_defaults(parser, str(path))

    assert "'help'" not in capsys.readouterr().err


def test_an_unknown_key_is_dropped_silently_when_not_strict(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("type: example_tonality\nnot_a_real_flag: 1\n")
    parser = _parser()

    config.apply_config_defaults(parser, str(path), strict=False)

    assert parser.parse_args([]).type == "example_tonality"


# --------------------------------------------------------------------------
# find_default_config_path
# --------------------------------------------------------------------------


def test_finds_a_config_json_in_the_default_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_DEFAULT_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.json").write_text("{}")

    assert config.find_default_config_path() == str(tmp_path / "config.json")


def test_a_yml_extension_is_recognised_too(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_DEFAULT_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yml").write_text("")

    assert config.find_default_config_path() == str(tmp_path / "config.yml")


def test_returns_none_when_no_config_file_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_DEFAULT_CONFIG_DIR", str(tmp_path))

    assert config.find_default_config_path() is None


def test_the_alphabetically_first_match_wins_when_several_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_DEFAULT_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "config.yaml").write_text("")

    assert config.find_default_config_path() == str(tmp_path / "config.json")


# --------------------------------------------------------------------------
# apply_config_file: --config, or an auto-discovered file, or neither. Both
# entry points resolve their config through this one function, so these
# precedence rules are pinned once here rather than per CLI.
# --------------------------------------------------------------------------


def _write_config(path, **values) -> str:
    path.write_text(json.dumps(values))
    return str(path)


def test_an_explicit_config_file_overrides_the_built_in_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog"])
    parser = _cli_parser()
    parser.set_defaults(config=_write_config(tmp_path / "my_config.json", epochs=99))

    assert config.apply_config_file(parser) is not None
    assert parser.parse_args([]).epochs == 99


def test_an_explicit_cli_flag_still_wins_over_the_config_file(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog"])
    parser = _cli_parser()
    parser.set_defaults(config=_write_config(tmp_path / "my_config.json", epochs=99))

    config.apply_config_file(parser)

    assert parser.parse_args(["--epochs", "3"]).epochs == 3


def test_no_config_flag_falls_back_to_an_auto_discovered_file(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog"])
    found = _write_config(tmp_path / "config.json", epochs=7)
    monkeypatch.setattr(config, "find_default_config_path", lambda: found)
    parser = _cli_parser()

    assert config.apply_config_file(parser) == found
    assert parser.parse_args([]).epochs == 7


def test_an_explicit_config_file_wins_over_an_auto_discovered_one(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog"])
    found = _write_config(tmp_path / "config.json", epochs=7)
    monkeypatch.setattr(config, "find_default_config_path", lambda: found)
    parser = _cli_parser()
    parser.set_defaults(config=_write_config(tmp_path / "my_config.json", epochs=99))

    config.apply_config_file(parser)

    assert parser.parse_args([]).epochs == 99


def test_no_config_anywhere_leaves_the_built_in_defaults_alone(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog"])
    monkeypatch.setattr(config, "find_default_config_path", lambda: None)
    parser = _cli_parser()

    assert config.apply_config_file(parser) is None
    assert parser.parse_args([]).epochs == 4


def test_an_auto_discovered_file_with_an_unknown_key_is_ignored_not_fatal(tmp_path, monkeypatch):
    # strict=False for the auto-discovered path: the same file is shared by
    # both entry points, so a key only the other one defines - or a stale one
    # - must not block every run. An explicit --config stays strict.
    monkeypatch.setattr("sys.argv", ["prog"])
    monkeypatch.setattr(
        config,
        "find_default_config_path",
        lambda: _write_config(tmp_path / "config.json", epochs=7, not_a_real_arg=True),
    )
    parser = _cli_parser()

    config.apply_config_file(parser)

    assert parser.parse_args([]).epochs == 7


def test_an_explicit_config_file_with_an_unknown_key_is_fatal(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["prog"])
    parser = _cli_parser()
    parser.set_defaults(config=_write_config(tmp_path / "my_config.json", not_a_real_arg=True))

    with pytest.raises(SystemExit):
        config.apply_config_file(parser)

    assert "not_a_real_arg" in capsys.readouterr().err


# --------------------------------------------------------------------------
# parse_with_sources: the whole resolution, plus where each value came from.
# --------------------------------------------------------------------------


def test_with_nothing_given_every_tracked_value_reads_as_a_default(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog"])
    monkeypatch.setattr(config, "find_default_config_path", lambda: None)

    args, sources, config_path = config.parse_with_sources(_cli_parser(), ("type", "epochs"))

    assert (args.type, args.epochs) == ("tonality", 4)
    assert sources == {"type": "default", "epochs": "default"}
    assert config_path is None


def test_a_value_taken_from_a_config_file_is_reported_as_config(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog"])
    monkeypatch.setattr(config, "find_default_config_path", lambda: None)
    given = _write_config(tmp_path / "my_config.json", epochs=99)
    monkeypatch.setattr("sys.argv", ["prog", "--config", given])

    args, sources, config_path = config.parse_with_sources(_cli_parser(), ("type", "epochs"))

    assert args.epochs == 99
    assert sources == {"type": "default", "epochs": "config"}
    assert config_path == given


def test_a_value_typed_on_the_command_line_is_reported_as_a_flag(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--epochs", "3"])
    monkeypatch.setattr(config, "find_default_config_path", lambda: None)

    args, sources, _ = config.parse_with_sources(_cli_parser(), ("type", "epochs"))

    assert args.epochs == 3
    assert sources == {"type": "default", "epochs": "flag"}


def test_a_flag_is_reported_as_a_flag_even_when_it_matches_the_config_value(tmp_path, monkeypatch):
    # The reason explicit_cli_dests() probes with a sentinel instead of
    # comparing values: these two agree, and the flag still has to win.
    given = _write_config(tmp_path / "my_config.json", epochs=99)
    monkeypatch.setattr("sys.argv", ["prog", "--config", given, "--epochs", "99"])
    monkeypatch.setattr(config, "find_default_config_path", lambda: None)

    _, sources, _ = config.parse_with_sources(_cli_parser(), ("epochs",))

    assert sources == {"epochs": "flag"}


def test_an_untracked_dest_gets_no_source_entry(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--epochs", "3"])
    monkeypatch.setattr(config, "find_default_config_path", lambda: None)

    _, sources, _ = config.parse_with_sources(_cli_parser(), ("type",))

    assert sources == {"type": "default"}
