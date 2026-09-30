"""Failure-path tests for Corpus.load.

test_corpus.py covers the validation rules that were already pinned. This
file covers the branches that were not reached at all: the two I/O guards in
load() itself, the three `_check_case` type guards, and - most importantly -
what the corpus holds *after* a load has failed.

Every one of these paths ends in utils.errors.fail, which prints and calls
sys.exit(1), so the assertion is always SystemExit rather than a domain
exception. That is the behaviour under review, not a property these tests
endorse; see the review notes on turning fail() into a raised error.
"""

import pytest

from chain_checker.baseclasses.corpus.corpus import Corpus

_VALID_TWO_CASES = """\
cases:
  - id: 1
    input:
      text: "hello"
    output:
      passed: true
  - id: 2
    input:
      text: "world"
    output:
      passed: false
"""


def _write(tmp_path, text: str) -> str:
    # tmp_path is pytest's own per-test directory, so there is nothing to
    # clean up and no delete=False dance as in test_corpus.py's _load.
    path = tmp_path / "corpus.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _load_expecting_exit(tmp_path, text: str) -> None:
    corpus = Corpus()
    with pytest.raises(SystemExit):
        corpus.load(_write(tmp_path, text))


# --------------------------------------------------------------------------
# load()'s own I/O guards
# --------------------------------------------------------------------------


def test_a_missing_corpus_file_exits_instead_of_raising_oserror(tmp_path):
    corpus = Corpus()

    with pytest.raises(SystemExit):
        corpus.load(str(tmp_path / "does-not-exist.yaml"))


def test_a_directory_passed_as_a_corpus_file_exits(tmp_path):
    # IsADirectoryError is an OSError, so it takes the same guard as a
    # missing file rather than escaping as a traceback.
    corpus = Corpus()

    with pytest.raises(SystemExit):
        corpus.load(str(tmp_path))


def test_malformed_yaml_exits(tmp_path):
    _load_expecting_exit(tmp_path, "cases: [unclosed\n")


def test_an_empty_file_exits(tmp_path):
    # yaml.load returns None for an empty document, which is not a dict and
    # so trips the missing-'cases' guard rather than an AttributeError.
    _load_expecting_exit(tmp_path, "")


def test_a_top_level_list_instead_of_a_mapping_exits(tmp_path):
    _load_expecting_exit(tmp_path, "- id: 1\n- id: 2\n")


def test_an_empty_cases_list_exits(tmp_path):
    # _check_file_data fails on this rather than returning [] - load() has no
    # branch left that treats an empty case list as anything but rejected.
    _load_expecting_exit(tmp_path, "cases: []\n")


# --------------------------------------------------------------------------
# _check_case type guards that no existing test reached
# --------------------------------------------------------------------------


def test_a_case_that_is_not_a_mapping_is_rejected(tmp_path):
    _load_expecting_exit(tmp_path, "cases:\n  - just-a-string\n")


def test_a_non_mapping_input_is_rejected(tmp_path):
    _load_expecting_exit(
        tmp_path,
        """\
cases:
  - id: 1
    input: not-a-mapping
    output:
      passed: true
""",
    )


def test_a_non_mapping_output_is_rejected(tmp_path):
    _load_expecting_exit(
        tmp_path,
        """\
cases:
  - id: 1
    input:
      text: "hello"
    output: not-a-mapping
""",
    )


def test_an_unquoted_numeric_output_field_name_is_rejected(tmp_path):
    # Record hands its field names back as list[str] and the report believes
    # it: rule names are joined into text and metrics.json sorts them. YAML
    # reads a bare `1` as an int, so the annotation stops being true without
    # anyone writing anything unusual.
    _load_expecting_exit(
        tmp_path,
        """\
cases:
  - id: 1
    input:
      text: "hello"
    output:
      1: true
""",
    )


def test_an_unquoted_boolean_output_field_name_is_rejected(tmp_path):
    # The nastier spelling: YAML reads a bare `no` as False, so a field named
    # for the Norwegian language check becomes a bool key.
    _load_expecting_exit(
        tmp_path,
        """\
cases:
  - id: 1
    input:
      text: "hello"
    output:
      no: true
""",
    )


def test_an_unquoted_input_field_name_is_rejected_too(tmp_path):
    # Input is a Record as well, and its field names are matched against the
    # chain's InputSchema by name.
    _load_expecting_exit(
        tmp_path,
        """\
cases:
  - id: 1
    input:
      1: "hello"
    output:
      passed: true
""",
    )


def test_a_quoted_numeric_field_name_is_accepted(tmp_path):
    # The guard rejects the type, not the spelling - "1" is a usable name.
    corpus = Corpus()
    corpus.load(
        _write(
            tmp_path,
            """\
cases:
  - id: 1
    input:
      "1": "hello"
    output:
      "2": true
""",
        )
    )

    assert corpus.get_by_id(1).get_output().get_keys() == ["2"]


def test_a_null_input_is_rejected(tmp_path):
    # A bare `input:` key parses to None. Without the isinstance guard this
    # would reach Input(None) and only fail later inside dict(None).
    _load_expecting_exit(
        tmp_path,
        """\
cases:
  - id: 1
    input:
    output:
      passed: true
""",
    )


def test_an_empty_input_mapping_is_rejected(tmp_path):
    # Symmetric with the empty-output guard now: an empty `output` can never
    # fail, and an empty `input` gives the chain nothing to run on. This used
    # to be accepted while empty output was rejected.
    _load_expecting_exit(
        tmp_path,
        """\
cases:
  - id: 1
    input: {}
    output:
      passed: true
""",
    )


# --------------------------------------------------------------------------
# Case ids have to survive being a file name, not just being a dict key
# --------------------------------------------------------------------------


def test_ids_that_differ_only_as_values_but_not_as_text_are_rejected(tmp_path):
    # `1` and `"1"` hash differently, so the duplicate-id check never saw
    # them - but loop/pathHandling.py keys the run cache by str(id) and names
    # each file f"{id}.txt", so both wrote and recalled the same '1.txt' and
    # one case silently got the other's cached prediction.
    _load_expecting_exit(
        tmp_path,
        """\
cases:
  - id: 1
    input:
      text: "a"
    output:
      passed: true
  - id: "1"
    input:
      text: "b"
    output:
      passed: false
""",
    )


def test_a_null_id_is_rejected(tmp_path):
    # A bare `id:` parses to None, which is hashable and so used to pass the
    # only id check there was - and then named a cache file 'None.txt'.
    _load_expecting_exit(
        tmp_path,
        """\
cases:
  - id:
    input:
      text: "hello"
    output:
      passed: true
""",
    )


def test_an_id_carrying_a_path_separator_is_rejected(tmp_path):
    # _save_prediction joins the id into the entries directory, so this used
    # to write outside the run directory entirely.
    _load_expecting_exit(
        tmp_path,
        """\
cases:
  - id: "../../pwn"
    input:
      text: "hello"
    output:
      passed: true
""",
    )


def test_an_empty_string_id_is_rejected(tmp_path):
    _load_expecting_exit(
        tmp_path,
        """\
cases:
  - id: ""
    input:
      text: "hello"
    output:
      passed: true
""",
    )


def test_an_ordinary_string_id_is_still_accepted(tmp_path):
    # The guard must not reject the normal case: ids are commonly slugs.
    corpus = Corpus()
    corpus.load(
        _write(
            tmp_path,
            """\
cases:
  - id: greeting-01
    input:
      text: "hello"
    output:
      passed: true
""",
        )
    )

    assert corpus.get_by_id("greeting-01").get_input().get() == {"text": "hello"}


# --------------------------------------------------------------------------
# Corpus state after a failed load
# --------------------------------------------------------------------------


def test_a_case_level_failure_leaves_no_partial_entries_behind(tmp_path):
    # load() builds a local dict and swaps it in only after every case has
    # passed, so a failure on case #2 does not leave case #1 behind. It used
    # to clear _entries up front and add as it validated, which left half a
    # corpus for any caller that recovered from the failure.
    corpus = Corpus()

    with pytest.raises(SystemExit):
        corpus.load(
            _write(
                tmp_path,
                """\
cases:
  - id: 1
    input:
      text: "would have been loaded before the bad case"
    output:
      passed: true
  - id: 2
    input: not-a-mapping
    output:
      passed: true
""",
            )
        )

    assert len(corpus) == 0


def test_a_failed_reload_keeps_the_previously_loaded_corpus(tmp_path):
    # A reload that fails on its first case used to destroy the corpus that
    # was already loaded and working. The swap now happens last, so the
    # working corpus survives a broken file.
    corpus = Corpus()
    corpus.load(_write(tmp_path, _VALID_TWO_CASES))
    assert len(corpus) == 2

    bad = tmp_path / "bad.yaml"
    bad.write_text("cases:\n  - just-a-string\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        corpus.load(str(bad))

    assert len(corpus) == 2
    assert corpus.get_by_id(1).get_input().get() == {"text": "hello"}


def test_a_top_level_failure_also_keeps_the_previously_loaded_corpus(tmp_path):
    # The file-level and case-level rejections now agree: same user error, a
    # broken corpus file, same outcome for the in-memory corpus. These two
    # used to diverge because only the case-level path cleared _entries.
    corpus = Corpus()
    corpus.load(_write(tmp_path, _VALID_TWO_CASES))

    bad = tmp_path / "bad.yaml"
    bad.write_text("not_cases: []\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        corpus.load(str(bad))

    assert len(corpus) == 2
