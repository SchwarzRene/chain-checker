import os

import pytest
from pydantic import BaseModel

from chain_checker.baseclasses.corpus.corpus import Corpus
from chain_checker.baseclasses.corpus.entry import Entry
from chain_checker.baseclasses.corpus.input import Input
from chain_checker.baseclasses.corpus.label import Label
from chain_checker.baseclasses.corpus.output_empty import EmptyModelOutput
from chain_checker.utils.checker_utils.validation import (
    find_corpus_path,
    validate_corpus,
)


class _InputSchema(BaseModel):
    text: str
    count: int = 0


class _OutputSchema(BaseModel):
    tonality: str


class _FakeChain:
    def __init__(self, input_schema=_InputSchema, output_schema=_OutputSchema) -> None:
        self.InputSchema = input_schema
        if output_schema is not None:
            self.OutputSchema = output_schema


class _FakeModel:
    def __init__(self, chain, chain_type: str = "template_checklist") -> None:
        self.chain = chain
        self._chain_type = chain_type

    def get_chain_type(self) -> str:
        return self._chain_type


def _entry(entry_id: str, input_data: dict, output_data: dict) -> Entry:
    return Entry(entry_id, Input(input_data), Label(output_data), {}, EmptyModelOutput())


def _corpus(*entries: Entry) -> Corpus:
    corpus = Corpus()
    for entry in entries:
        corpus.add_entry(entry)
    return corpus


# --------------------------------------------------------------------------
# find_corpus_path
# --------------------------------------------------------------------------


def test_the_single_yaml_file_in_the_type_dir_is_found(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workflows" / "tonality").mkdir(parents=True)
    (tmp_path / "workflows" / "tonality" / "corpus.yaml").write_text("cases: []\n")

    # glob() returns a path relative to cwd, matching the relative pattern
    # find_corpus_path() searched with - not an absolute path.
    assert find_corpus_path("tonality") == os.path.join("workflows", "tonality", "corpus.yaml")


def test_no_yaml_file_fails_loudly(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workflows" / "tonality").mkdir(parents=True)

    with pytest.raises(SystemExit):
        find_corpus_path("tonality")

    assert "No .yaml corpus file found" in capsys.readouterr().err


def test_several_yaml_files_fail_with_the_type_named(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    type_dir = tmp_path / "workflows" / "tonality"
    type_dir.mkdir(parents=True)
    (type_dir / "a.yaml").write_text("cases: []\n")
    (type_dir / "b.yaml").write_text("cases: []\n")

    with pytest.raises(SystemExit):
        find_corpus_path("tonality")

    assert "Multiple .yaml corpus files found" in capsys.readouterr().err


# --------------------------------------------------------------------------
# validate_corpus
# --------------------------------------------------------------------------


def test_matching_input_and_output_keys_pass_without_raising():
    corpus = _corpus(_entry("a", {"text": "hi"}, {"tonality": "formal"}))

    validate_corpus(_FakeModel(_FakeChain()), corpus)


def test_a_chain_with_no_outputschema_fails_loudly(capsys):
    corpus = _corpus(_entry("a", {"text": "hi"}, {"tonality": "formal"}))
    chain = _FakeChain(output_schema=None)

    with pytest.raises(SystemExit):
        validate_corpus(_FakeModel(chain), corpus)

    assert "has no OutputSchema" in capsys.readouterr().err


def test_an_input_key_the_schema_does_not_have_fails_with_the_key_named(capsys):
    corpus = _corpus(_entry("a", {"text": "hi", "extra": "nope"}, {"tonality": "formal"}))

    with pytest.raises(SystemExit):
        validate_corpus(_FakeModel(_FakeChain()), corpus)

    err = capsys.readouterr().err
    assert "Unknown key(s): ['extra']" in err
    assert "case 'a'" in err


def test_a_missing_required_input_field_fails_with_the_field_named(capsys):
    corpus = _corpus(_entry("a", {}, {"tonality": "formal"}))

    with pytest.raises(SystemExit):
        validate_corpus(_FakeModel(_FakeChain()), corpus)

    assert "Missing required key(s): ['text']" in capsys.readouterr().err


def test_an_input_value_the_schema_rejects_fails_with_the_pydantic_reason(capsys):
    corpus = _corpus(_entry("a", {"text": "hi", "count": "not-a-number"}, {"tonality": "formal"}))

    with pytest.raises(SystemExit):
        validate_corpus(_FakeModel(_FakeChain()), corpus)

    err = capsys.readouterr().err
    assert "does not match chain 'template_checklist's InputSchema" in err
    assert "Reason:" in err


def test_an_output_key_the_outputschema_does_not_have_fails_with_the_key_named(capsys):
    corpus = _corpus(_entry("a", {"text": "hi"}, {"tonality": "formal", "unknown_field": "x"}))

    with pytest.raises(SystemExit):
        validate_corpus(_FakeModel(_FakeChain()), corpus)

    assert "Unknown: ['unknown_field']" in capsys.readouterr().err
