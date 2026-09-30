"""Do the metrics agree with Label.matches() about what "correct" means?

They have to. Two things grade a run - a graded PASS/FAIL, via
Label.matches(), and the epoch report's accuracy, via SingleValueMetrics -
and if they can disagree then one corpus case is simultaneously a
counter-example handed to the prompt modifier and a correct answer in the
number the training loop optimizes.

Every metric routes through corpus/label.py's values_agree, the same function
matches() calls, so this file is a specification: each row asserts that both
graders reach the same verdict.
"""

import pytest

from chain_checker.baseclasses.corpus.corpus import Corpus
from chain_checker.baseclasses.corpus.label import Label
from chain_checker.baseclasses.metrics.m_mispredicted import MispredictedMetrics
from chain_checker.baseclasses.metrics.m_single_value import (
    SingleValueMetrics,
    iter_rule_results,
)


def _matches_says_correct(expected: dict, predicted: dict) -> bool:
    # What a graded PASS/FAIL is printed from, and what MispredictedMetrics
    # records a counter-example from.
    return all(Label(expected).matches(Label(predicted)).values())


def _accuracy_says_correct(make_entry, expected: dict, predicted: dict) -> bool:
    # What the epoch report's headline accuracy is built from - except a
    # numeric ("mae") rule, which the headline deliberately excludes (tracked
    # by its own MAE chart instead, see
    # SingleValueMetrics._calculate_mean_accuracy). For those, the rule's own
    # accuracy - the same within-tolerance match rate - is the comparable verdict.
    metric = SingleValueMetrics("Accuracy-Metrics")
    metric.add_entry(make_entry("a", expected=expected, predicted=predicted))
    results = metric.compute()
    for _, result in iter_rule_results(results):
        if "mae" in result:
            return result["accuracy"] == 1.0
    return results["accuracy"] == 1.0


def _is_a_counter_example(make_entry, expected: dict, predicted: dict) -> bool:
    metric = MispredictedMetrics()
    metric.add_entry(make_entry("a", expected=expected, predicted=predicted))
    return "a" in metric.compute()


# Every row is (expected, predicted, is-correct). The three graders are
# asserted against the same expectation, so a change to any one of them shows
# up here rather than as a quiet drift between two reports.
_CASES = [
    # Ordinary bools - these always agreed.
    ({"passed": True}, {"passed": True}, True),
    ({"passed": True}, {"passed": False}, False),
    ({"passed": False}, {"passed": False}, True),
    ({"passed": True}, {}, False),
    ({"verdicts": {"r1": True}}, {"verdicts": {"r1": True}}, True),
    ({"verdicts": {"r1": True}}, {"verdicts": {"r1": False}}, False),
    ({"verdicts": {"r1": True}}, {"verdicts": {}}, False),
    # An unexpected sub-key: values_agree needs the exact key set, so the whole
    # key fails. Crediting the sub-rules that line up reports a flat 1.0 here.
    ({"verdicts": {"r1": True}}, {"verdicts": {"r1": True, "r2": True}}, False),
    # Mistyped truthy predictions: right under truthiness, wrong to the grader.
    ({"passed": True}, {"passed": 1}, False),
    ({"passed": True}, {"passed": "yes"}, False),
    ({"passed": True}, {"passed": [0]}, False),
    ({"verdicts": {"r1": True}}, {"verdicts": {"r1": 1}}, False),
    # Mistyped falsy predictions - the more dangerous direction, where
    # truthiness reads the chain saying nothing as a correct `false`.
    ({"passed": False}, {"passed": 0}, False),
    ({"passed": False}, {"passed": None}, False),
    ({"passed": False}, {"passed": ""}, False),
    ({"passed": False}, {}, False),
    # Numbers, including the float tolerance matches() applies.
    ({"score": 0.3}, {"score": 0.1 + 0.2}, True),
    # An expected 0.0 needs isclose's abs_tol; rel_tol alone is 0 there.
    ({"score": 0.0}, {"score": 1e-12}, True),
    ({"score": 0.0}, {"score": 1e-6}, False),
    ({"score": 0.9}, {"score": 0.1}, False),
    ({"score": 0.5}, {"score": "0.5"}, False),
    ({"score": 1.0}, {}, False),
    # A key the corpus expects to be null. Reading an absent key as None would
    # credit the second row, which matches() fails.
    ({"nothing": None}, {"nothing": None}, True),
    ({"nothing": None}, {}, False),
    # Values that are neither bool nor number: SingleValueMetrics scores them
    # by agreement, on the terms matches() grades them.
    ({"echo": "Hello there"}, {"echo": "Hello there"}, True),
    ({"echo": "Hello there"}, {"echo": "something else"}, False),
    ({"v": {"value": "ok", "note": "why"}}, {"v": {"value": "ok", "note": "why"}}, True),
    ({"v": {"value": "ok", "note": "why"}}, {"v": {"value": "no", "note": "why"}}, False),
    ({"tags": [1, 2]}, {"tags": [1, 2]}, True),
    ({"tags": [1, 2]}, {"tags": [2, 1]}, False),
]

_IDS = [
    "true-true",
    "true-false",
    "false-false",
    "key-missing",
    "nested-true-true",
    "nested-true-false",
    "nested-key-missing",
    "nested-extra-key",
    "int-1",
    "truthy-string",
    "truthy-list",
    "nested-int-1",
    "int-0",
    "none",
    "empty-string",
    "false-key-missing",
    "float-tolerance",
    "float-zero-tolerance",
    "float-zero-real-diff",
    "float-wrong",
    "float-as-string",
    "float-missing",
    "null-expected-null-given",
    "null-expected-key-missing",
    "str-same",
    "str-diff",
    "dict-same",
    "dict-diff",
    "list-same",
    "list-diff",
]


@pytest.mark.parametrize(("expected", "predicted", "correct"), _CASES, ids=_IDS)
def test_the_report_agrees_with_checker_about_every_case(make_entry, expected, predicted, correct):
    # The invariant this file exists for: the number the training loop
    # optimizes and the verdict a human reads off the grader cannot disagree.
    assert _matches_says_correct(expected, predicted) is correct
    assert _accuracy_says_correct(make_entry, expected, predicted) is correct


@pytest.mark.parametrize(("expected", "predicted", "correct"), _CASES, ids=_IDS)
def test_a_case_is_a_counter_example_exactly_when_it_is_wrong(
    make_entry, expected, predicted, correct
):
    # And the evidence handed to the prompt modifier is the complement of the
    # accuracy, rather than a separate opinion about the same entry.
    assert _is_a_counter_example(make_entry, expected, predicted) is (not correct)


# --------------------------------------------------------------------------
# Where the two are answering genuinely different questions
# --------------------------------------------------------------------------


def test_mae_measures_distance_while_accuracy_measures_agreement(make_entry):
    # Not a divergence - pinned so it is not "fixed" by accident. matches()
    # asks "is this right" with isclose; 'mae' asks "how far off".
    expected, predicted = {"score": 0.3}, {"score": 0.1 + 0.2}

    assert _matches_says_correct(expected, predicted) is True

    metric = SingleValueMetrics("Accuracy-Metrics")
    metric.add_entry(make_entry("a", expected=expected, predicted=predicted))
    score = metric.compute()["rules"]["score"]["results"]

    assert score["mae"] == pytest.approx(0.0, abs=1e-9)
    assert score["accuracy"] == 1.0


# --------------------------------------------------------------------------
# The two type-consistency checks word one rejection one way
# --------------------------------------------------------------------------


def _rejected_at_load(tmp_path, first: str, second: str) -> str:
    corpus_file = tmp_path / "corpus.yaml"
    corpus_file.write_text(
        "cases:\n"
        "  - id: case-0\n"
        "    input: {text: t}\n"
        f"    output: {{{first}}}\n"
        "  - id: case-1\n"
        "    input: {text: t}\n"
        f"    output: {{{second}}}\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as raised:
        Corpus().load(str(corpus_file))
    return str(raised.value)


def _rejected_by_the_metric(make_entry, first: dict, second: dict) -> str:
    metric = SingleValueMetrics("Accuracy-Metrics")
    metric.add_entry(make_entry("case-0", expected=first, predicted=first))

    with pytest.raises(SystemExit) as raised:
        metric.add_entry(make_entry("case-1", expected=second, predicted=second))
    return str(raised.value)


def test_both_type_checks_word_the_same_rejection_identically(tmp_path, make_entry):
    # Corpus.load() rejects an inconsistent key before a model call; the metric
    # is the backstop for a corpus assembled through add_entry(). One rejection,
    # so the text has one owner - the load-time one only adds the file it read.
    from_load = _rejected_at_load(tmp_path, "score: 0.9", "score: high")
    from_metric = _rejected_by_the_metric(make_entry, {"score": 0.9}, {"score": "high"})

    shared = "key 'score' changes type between cases"
    assert from_load[from_load.index(shared) :] == from_metric[from_metric.index(shared) :]
    assert "Corpus file" in from_load
    assert "case-0" in from_metric and "case-1" in from_metric


def test_the_two_type_checks_reject_the_same_corpora(tmp_path, make_entry):
    # Agreeing on the wording is worth nothing if they disagree on what to
    # reject, so the same pairs go through both.
    pairs = [
        ("passed: true", 'passed: "true"', {"passed": True}, {"passed": "true"}),
        ("score: 3", "score: true", {"score": 3}, {"score": True}),
        ("verdicts: {}", "verdicts: {r1: true}", {"verdicts": {}}, {"verdicts": {"r1": True}}),
    ]

    for first_yaml, second_yaml, first, second in pairs:
        assert "changes type between cases" in _rejected_at_load(tmp_path, first_yaml, second_yaml)
        assert "changes type between cases" in _rejected_by_the_metric(make_entry, first, second)


def test_a_dict_of_bools_is_expanded_while_matches_grades_it_whole(make_entry):
    # The one structural difference, by design: matches() reports one verdict
    # for the key, the metric one per sub-rule so the report can show which
    # rule the chain struggles with. They still agree the entry was wrong.
    expected = {"verdicts": {"r1": True, "r2": True}}
    predicted = {"verdicts": {"r1": True, "r2": False}}

    assert _matches_says_correct(expected, predicted) is False

    metric = SingleValueMetrics("Accuracy-Metrics")
    metric.add_entry(make_entry("a", expected=expected, predicted=predicted))
    results = metric.compute()
    sub_rules = results["rules"]["verdicts"]["sub_rules"]

    assert sub_rules["r1"]["results"]["accuracy"] == 1.0
    assert sub_rules["r2"]["results"]["accuracy"] == 0.0
    assert results["accuracy"] == pytest.approx(0.5)
