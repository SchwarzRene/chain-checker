"""Unit tests for chain_checker/baseclasses/corpus/corpus.py."""

import os
import tempfile

import pytest

from chain_checker.baseclasses.corpus.corpus import Corpus
from chain_checker.baseclasses.corpus.entry import Entry
from chain_checker.baseclasses.corpus.input import Input
from chain_checker.baseclasses.corpus.label import Label
from chain_checker.baseclasses.corpus.output_empty import EmptyModelOutput

_VALID_CORPUS = """\
cases:
  - id: 1
    input:
      text: "hello"
    output:
      passed: true
    info:
      language: en
  - id: 2
    input:
      text: "world"
    output:
      passed: false
"""


def _load(text: str) -> Corpus:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(text)
        path = f.name
    try:
        corpus = Corpus()
        corpus.load(path)
        return corpus
    finally:
        os.unlink(path)


def _load_expecting_failure(text: str) -> None:
    with pytest.raises(SystemExit):
        _load(text)


def test_load_reads_every_case():
    corpus = _load(_VALID_CORPUS)

    assert len(corpus) == 2
    assert corpus.get_by_id(1).get_input().get() == {"text": "hello"}
    assert corpus.get_by_id(2).get_output().get() == {"passed": False}


def test_case_without_info_gets_an_empty_dict_not_a_crash():
    corpus = _load(_VALID_CORPUS)

    assert corpus.get_by_id(2).get_info("language") is None


def test_case_with_explicit_null_info_gets_an_empty_dict_not_a_crash():
    # A bare `info:` key (no value) parses to None via PyYAML - this used to
    # make Entry._info None, and entry.get_info(...) would then crash with
    # AttributeError: 'NoneType' object has no attribute 'get'
    corpus = _load("""\
cases:
  - id: 1
    input:
      text: "hello"
    output:
      passed: true
    info:
""")

    assert corpus.get_by_id(1).get_info("language") is None


def test_iterating_the_corpus_visits_every_entry_exactly_once():
    corpus = _load(_VALID_CORPUS)

    ids = sorted(entry.get_id() for entry in corpus)

    assert ids == [1, 2]


def test_reloading_a_corpus_replaces_its_entries_instead_of_merging():
    corpus = Corpus()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(_VALID_CORPUS)
        first_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write("""\
cases:
  - id: 99
    input:
      text: "only this one"
    output:
      passed: true
""")
        second_path = f.name

    try:
        corpus.load(first_path)
        corpus.load(second_path)

        assert len(corpus) == 1
        assert corpus.get_by_id(99) is not None
    finally:
        os.unlink(first_path)
        os.unlink(second_path)


def test_empty_cases_list_is_rejected():
    _load_expecting_failure("cases: []\n")


def test_cases_not_a_list_is_rejected():
    _load_expecting_failure("cases: not-a-list\n")


def test_missing_top_level_cases_key_is_rejected():
    _load_expecting_failure("not_cases: []\n")


def test_case_missing_a_required_key_is_rejected():
    _load_expecting_failure("""\
cases:
  - id: 1
    input:
      text: "hello"
""")


def test_case_with_empty_output_is_rejected():
    _load_expecting_failure("""\
cases:
  - id: 1
    input:
      text: "hello"
    output: {}
""")


def test_duplicate_case_id_is_rejected():
    _load_expecting_failure("""\
cases:
  - id: 1
    input:
      text: "a"
    output:
      passed: true
  - id: 1
    input:
      text: "b"
    output:
      passed: false
""")


def test_unhashable_case_id_is_rejected():
    _load_expecting_failure("""\
cases:
  - id: [1, 2]
    input:
      text: "hello"
    output:
      passed: true
""")


def test_non_mapping_info_is_rejected():
    _load_expecting_failure("""\
cases:
  - id: 1
    input:
      text: "hello"
    output:
      passed: true
    info: "not a mapping"
""")


def _make_entry(entry_id) -> Entry:
    return Entry(
        entry_id,
        Input({"text": "hello"}),
        Label({"passed": True}),
        {},
        EmptyModelOutput(),
    )


def test_add_entry_rejects_a_duplicate_id():
    corpus = Corpus()
    corpus.add_entry(_make_entry(1))

    with pytest.raises(ValueError):
        corpus.add_entry(_make_entry(1))


def test_get_by_idx_raises_instead_of_wrapping_past_the_end():
    # This used to be `idx % len(self)`, so index 2 on a 2-entry corpus
    # silently returned entry 0 and an off-by-one re-scored the wrong case.
    corpus = _load(_VALID_CORPUS)

    assert corpus.get_by_idx(0).get_id() == 1
    assert corpus.get_by_idx(1).get_id() == 2

    with pytest.raises(IndexError):
        corpus.get_by_idx(2)


def test_get_by_idx_on_an_empty_corpus_raises():
    corpus = Corpus()

    with pytest.raises(IndexError):
        corpus.get_by_idx(0)


def test_reset_replaces_every_entrys_model_output_with_an_empty_one():
    corpus = _load(_VALID_CORPUS)
    entry = corpus.get_by_id(1)
    entry.get_model_output().set_output({"passed": True})

    corpus.reset()

    assert corpus.get_by_id(1).get_model_output().get() == {}


def test_iter_returns_a_real_iterator_over_every_entry():
    # Was a hand-rolled CorpusIterator class; __iter__ now returns
    # iter(list(...)), which keeps the snapshot behaviour the metrics rely on
    # (see test_corpus_contracts.py) without reimplementing the protocol.
    corpus = _load(_VALID_CORPUS)

    iterator = iter(corpus)

    assert iter(iterator) is iterator
    assert sorted(entry.get_id() for entry in list(iterator)) == [1, 2]
