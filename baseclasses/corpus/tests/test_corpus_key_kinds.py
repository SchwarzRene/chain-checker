"""Load-time tests for Corpus._check_key_kinds.

A key scored as a number and a key scored as a string cannot share one
metric, so every case that checks a key has to agree on what kind of value it
holds. That disagreement is a property of the .yaml, so it is rejected while
loading - before any tool has spent anything on the model, and for every tool
that loads a corpus rather than only the one that builds metrics.

Cross-case, unlike the guards in _check_case, so these all need two cases.
"""

import pytest

from chain_checker.baseclasses.corpus.corpus import Corpus


def _corpus_text(*outputs: str) -> str:
    cases = "".join(
        f'  - id: case-{i}\n    input:\n      text: "t"\n    output:\n      {out}\n'
        for i, out in enumerate(outputs)
    )
    return f"cases:\n{cases}"


def _write(tmp_path, text: str) -> str:
    # mkdir because one test writes two corpora into sibling directories to
    # load one after the other.
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "corpus.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _load(tmp_path, *outputs: str) -> Corpus:
    corpus = Corpus()
    corpus.load(_write(tmp_path, _corpus_text(*outputs)))
    return corpus


def _load_expecting_exit(tmp_path, *outputs: str) -> str:
    corpus = Corpus()
    with pytest.raises(SystemExit) as exc:
        corpus.load(_write(tmp_path, _corpus_text(*outputs)))
    return str(exc.value)


# --------------------------------------------------------------------------
# Rejected at load
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("passed: true", 'passed: "true"'),
        ("passed: true", "passed: maybe"),
        ("score: 0.9", "score:"),
        ("score: 3", "score: true"),
        ("verdicts: {}", "verdicts: {r1: true}"),
        ("tag: en", "tag: 3"),
    ],
    ids=[
        "bool-then-quoted-bool",
        "bool-then-string",
        "number-then-null",
        "number-then-bool",
        "empty-mapping-then-populated",
        "string-then-number",
    ],
)
def test_a_key_whose_kind_changes_between_cases_is_rejected(tmp_path, first, second):
    message = _load_expecting_exit(tmp_path, first, second)

    assert "changes type between cases" in message


def test_the_order_of_the_two_cases_does_not_matter(tmp_path):
    # Whichever case establishes the kind, the other one disagrees with it.
    forwards = _load_expecting_exit(tmp_path, "score: 0.9", "score:")
    backwards = _load_expecting_exit(tmp_path, "score:", "score: 0.9")

    assert "changes type between cases" in forwards
    assert "changes type between cases" in backwards


def test_the_message_names_both_cases_and_both_values(tmp_path):
    # A locator for a human editing the .yaml: which two cases disagree, and
    # what each of them wrote.
    message = _load_expecting_exit(tmp_path, "score: 0.9", "score:")

    assert "'case-0'" in message
    assert "'case-1'" in message
    assert "0.9" in message
    assert "None" in message


def test_a_rejected_kind_leaves_the_previously_loaded_corpus_intact(tmp_path):
    # The check runs before the build-then-swap, like every other guard, so a
    # bad file must not half-replace a working corpus.
    corpus = Corpus()
    corpus.load(_write(tmp_path / "good", _corpus_text("passed: true")))
    assert len(corpus) == 1

    with pytest.raises(SystemExit):
        corpus.load(_write(tmp_path / "bad", _corpus_text("passed: true", "passed: 3")))

    assert len(corpus) == 1
    assert corpus.get_by_id("case-0").get_output().get_value("passed") is True


# --------------------------------------------------------------------------
# Accepted at load
# --------------------------------------------------------------------------


def test_a_key_holding_the_same_kind_throughout_loads(tmp_path):
    corpus = _load(tmp_path, "passed: true", "passed: false")

    assert len(corpus) == 2


def test_an_int_and_a_float_on_one_key_are_the_same_kind(tmp_path):
    # Both are numbers and both belong in the same metric, so a corpus mixing
    # 3 and 3.5 has to load.
    corpus = _load(tmp_path, "score: 3", "score: 3.5")

    assert len(corpus) == 2


def test_cases_need_not_check_the_same_keys(tmp_path):
    # A corpus may check different keys per case; only a key two cases *both*
    # check has to agree.
    corpus = _load(tmp_path, "passed: true", "score: 0.5")

    assert len(corpus) == 2


def test_a_key_that_is_null_in_every_case_loads(tmp_path):
    # Consistent, so it runs: `null` is a real assertion about the output.
    corpus = _load(tmp_path, "nothing:", "nothing:")

    assert len(corpus) == 2


def test_a_mapping_populated_in_every_case_loads(tmp_path):
    corpus = _load(tmp_path, "verdicts: {r1: true}", "verdicts: {r1: false}")

    assert len(corpus) == 2


def test_a_single_case_corpus_has_nothing_to_disagree_with(tmp_path):
    corpus = _load(tmp_path, "score: 0.9")

    assert len(corpus) == 1
