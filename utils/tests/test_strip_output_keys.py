import sys

import pytest
import yaml

from chain_checker.utils import strip_output_keys as strip_mod
from chain_checker.utils.strip_output_keys import strip_output_keys


def _corpus(*cases) -> dict:
    return {"cases": list(cases)}


# --------------------------------------------------------------------------
# strip_output_keys
# --------------------------------------------------------------------------


def test_the_named_key_is_removed_from_every_cases_output():
    data = _corpus(
        {"id": 1, "output": {"passed": True, "score": 0.9}},
        {"id": 2, "output": {"passed": False, "score": 0.1}},
    )

    result = strip_output_keys(data, ["score"], "corpus.yaml")

    assert result["cases"][0]["output"] == {"passed": True}
    assert result["cases"][1]["output"] == {"passed": False}


def test_several_keys_can_be_dropped_at_once():
    data = _corpus({"id": 1, "output": {"a": 1, "b": 2, "c": 3}})

    result = strip_output_keys(data, ["a", "b"], "corpus.yaml")

    assert result["cases"][0]["output"] == {"c": 3}


def test_a_key_the_case_never_had_is_ignored():
    data = _corpus({"id": 1, "output": {"passed": True}})

    result = strip_output_keys(data, ["score"], "corpus.yaml")

    assert result["cases"][0]["output"] == {"passed": True}


def test_a_case_whose_output_is_not_a_dict_is_left_alone():
    other_case = {"id": 2, "output": {"passed": True, "score": 1}}
    data = _corpus({"id": 1, "output": "not-a-dict"}, other_case)

    result = strip_output_keys(data, ["score"], "corpus.yaml")

    assert result["cases"][0]["output"] == "not-a-dict"


def test_a_non_dict_case_is_left_alone():
    data = _corpus("not-a-case", {"id": 1, "output": {"passed": True, "score": 1}})

    result = strip_output_keys(data, ["score"], "corpus.yaml")

    assert result["cases"][0] == "not-a-case"


def test_dropping_a_cases_only_key_fails_loudly(capsys):
    data = _corpus({"id": "c1", "output": {"passed": True}})

    with pytest.raises(SystemExit):
        strip_output_keys(data, ["passed"], "corpus.yaml")

    err = capsys.readouterr().err
    assert "empty output" in err
    assert "c1" in err


def test_missing_cases_list_fails_loudly(capsys):
    with pytest.raises(SystemExit):
        strip_output_keys({"not": "a corpus"}, ["score"], "corpus.yaml")

    assert "no top-level 'cases' list" in capsys.readouterr().err


def test_a_non_dict_top_level_value_also_fails():
    with pytest.raises(SystemExit):
        strip_output_keys(["not", "a", "dict"], ["score"], "corpus.yaml")


# --------------------------------------------------------------------------
# parse_args
# --------------------------------------------------------------------------


def test_parse_args_requires_file_out_and_drop(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["strip_output_keys.py"])

    with pytest.raises(SystemExit):
        strip_mod.parse_args()


def test_parse_args_reads_the_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "strip_output_keys.py",
            "--file",
            "in.yaml",
            "--out",
            "out.yaml",
            "--drop",
            "score",
            "passed",
        ],
    )

    args = strip_mod.parse_args()

    assert args.file == "in.yaml"
    assert args.out == "out.yaml"
    assert args.drop == ["score", "passed"]


# --------------------------------------------------------------------------
# main - the whole read/strip/write pipeline
# --------------------------------------------------------------------------


def _run_main(monkeypatch, *, file: str, out: str, drop: list[str]) -> None:
    monkeypatch.setattr(
        sys, "argv", ["strip_output_keys.py", "--file", file, "--out", out, "--drop", *drop]
    )
    strip_mod.main()


def test_main_reads_strips_and_writes_the_corpus(tmp_path, monkeypatch, capsys):
    src = tmp_path / "in.yaml"
    src.write_text("cases:\n  - id: 1\n    output:\n      passed: true\n      score: 0.9\n")
    out = tmp_path / "out.yaml"

    _run_main(monkeypatch, file=str(src), out=str(out), drop=["score"])

    assert yaml.safe_load(out.read_text()) == {"cases": [{"id": 1, "output": {"passed": True}}]}
    assert "Wrote" in capsys.readouterr().out


def test_a_multiline_kept_value_round_trips_through_the_dumper(tmp_path, monkeypatch):
    src = tmp_path / "in.yaml"
    src.write_text(
        'cases:\n  - id: 1\n    output:\n      passed: true\n      note: "line one\\nline two"\n'
    )
    out = tmp_path / "out.yaml"

    _run_main(monkeypatch, file=str(src), out=str(out), drop=["score"])

    written = yaml.safe_load(out.read_text())
    assert written["cases"][0]["output"]["note"] == "line one\nline two"


def test_main_fails_loudly_on_a_missing_source_file(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _run_main(
            monkeypatch,
            file=str(tmp_path / "missing.yaml"),
            out=str(tmp_path / "out.yaml"),
            drop=["score"],
        )

    assert "Could not open corpus file" in capsys.readouterr().err


def test_main_fails_loudly_on_invalid_yaml(tmp_path, monkeypatch, capsys):
    src = tmp_path / "bad.yaml"
    src.write_text("cases: [unterminated")

    with pytest.raises(SystemExit):
        _run_main(monkeypatch, file=str(src), out=str(tmp_path / "out.yaml"), drop=["score"])

    assert "is not valid YAML" in capsys.readouterr().err
