"""Unit tests for m_single_value.py.

SingleValueMetrics is the router: it looks at each key in a case's expected
output and picks the sub-metric that can score that type, expanding a
dict-of-bool into one boolean sub-metric per sub-key. The top-level
'accuracy' it reports is the run's headline number.

It grades by reading values out of the prediction directly, NOT through
Label.matches() - see test_matches_consistency.py for what that costs.
"""

import json

import pytest

from chain_checker.baseclasses.metrics.m_single_value import (
    SingleValueMetrics,
    iter_rule_results,
    rule_label,
)


def _flat(metric: SingleValueMetrics) -> dict:
    # compute()'s nested shape flattened to {rule-label: results, 'accuracy':
    # headline}, so an assertion about one rule stays a single line. The
    # nesting itself is asserted in the reported-shape section at the bottom.
    results = metric.compute()
    flat = {rule_label(path): sub for path, sub in iter_rule_results(results)}
    flat["accuracy"] = results["accuracy"]
    return flat


def _fed(make_entry, *cases) -> SingleValueMetrics:
    metric = SingleValueMetrics("Accuracy-Metrics")
    for i, (expected, predicted) in enumerate(cases):
        metric.add_entry(make_entry(f"case-{i}", expected=expected, predicted=predicted))
    return metric


# --------------------------------------------------------------------------
# Routing a key to the sub-metric that can score it
# --------------------------------------------------------------------------


def test_a_bool_key_is_scored_as_a_confusion_matrix(make_entry):
    metric = _fed(
        make_entry,
        ({"passed": True}, {"passed": True}),
        ({"passed": False}, {"passed": True}),
    )

    assert _flat(metric)["passed"] == {
        "accuracy": 0.5,
        "TT": 1,
        "TF": 0,
        "FT": 1,
        "FF": 0,
    }


def test_a_float_key_is_scored_as_a_mean_absolute_error(make_entry):
    metric = _fed(make_entry, ({"score": 1.0}, {"score": 0.5}))

    assert _flat(metric)["score"]["mae"] == pytest.approx(0.5)
    assert "TT" not in _flat(metric)["score"]


def test_an_int_key_is_routed_to_the_numeric_metric_not_the_boolean_one(make_entry):
    # bool is a subclass of int, so the order of the checks in value_kind is
    # load-bearing in both directions. Asserted through the reported shape:
    # 'mae' is a numeric metric's, TT/TF/FT/FF a boolean one's.
    metric = _fed(make_entry, ({"count": 3}, {"count": 5}))

    assert _flat(metric)["count"]["mae"] == pytest.approx(2.0)
    assert "TT" not in _flat(metric)["count"]


def test_a_bool_key_is_not_routed_to_the_numeric_metric(make_entry):
    metric = _fed(make_entry, ({"passed": True}, {"passed": True}))

    assert _flat(metric)["passed"]["TT"] == 1
    assert "mae" not in _flat(metric)["passed"]


def test_the_sub_metric_is_created_once_and_reused_across_entries(make_entry):
    # Otherwise each entry would reset the key's history and the run would
    # report the last entry's result as the whole corpus's.
    metric = _fed(
        make_entry,
        ({"passed": True}, {"passed": True}),
        ({"passed": True}, {"passed": True}),
        ({"passed": True}, {"passed": False}),
    )

    assert _flat(metric)["passed"]["TT"] == 2
    assert _flat(metric)["passed"]["TF"] == 1


def test_cases_checking_different_keys_each_get_their_own_sub_metric(make_entry):
    # A corpus need not check the same keys on every case.
    metric = _fed(
        make_entry,
        ({"passed": True}, {"passed": True}),
        ({"score": 1.0}, {"score": 1.0}),
    )

    assert set(_flat(metric)) == {"passed", "score", "accuracy"}


# --------------------------------------------------------------------------
# A missing or unusable prediction
# --------------------------------------------------------------------------


def test_a_key_the_chain_never_produced_is_a_mismatch_not_a_crash(make_entry):
    # Same rule Label.matches() uses: presence in the expected output is what
    # makes a key get checked.
    metric = _fed(make_entry, ({"passed": True}, {}))

    assert _flat(metric)["passed"]["TF"] == 1


def test_a_key_the_chain_never_produced_is_left_out_of_a_numeric_average(make_entry):
    metric = _fed(make_entry, ({"score": 1.0}, {}))

    assert _flat(metric)["score"]["mae"] == 0.0
    assert _flat(metric)["score"]["predicted_scores"] == [None]


def test_extra_keys_the_chain_volunteered_are_never_scored(make_entry):
    metric = _fed(make_entry, ({"passed": True}, {"passed": True, "reasoning": "because"}))

    assert set(_flat(metric)) == {"passed", "accuracy"}


def test_a_key_the_corpus_expects_to_be_null_is_not_credited_for_being_absent(make_entry):
    # Reading a missing key as None makes the two indistinguishable, scoring a
    # case that expects null at 1.0 while Label.matches() fails the same entry
    # - the one thing test_matches_consistency.py exists to rule out.
    metric = _fed(make_entry, ({"nothing": None}, {}))

    assert _flat(metric)["accuracy"] == 0.0
    assert _flat(metric)["nothing"]["matched"] == 0


def test_a_key_the_corpus_expects_to_be_null_still_agrees_when_the_chain_says_null(
    make_entry,
):
    # The other half: null is a value a corpus can legitimately check for.
    metric = _fed(make_entry, ({"nothing": None}, {"nothing": None}))

    assert _flat(metric)["accuracy"] == 1.0


def test_the_marker_for_an_absent_key_never_reaches_the_reported_results(make_entry):
    # A missing key is carried as a sentinel object rather than None. It has to
    # stay inside the metric: the results dict is written out as JSON, and
    # json.dumps raises on a bare object().
    metric = _fed(
        make_entry,
        (
            {"passed": True, "score": 1.0, "tag": "en", "verdicts": {"r1": True}},
            {},
        ),
    )

    json.dumps(metric.compute())


def test_a_missing_sub_key_is_not_credited_against_an_expected_null_either(make_entry):
    # A dict-of-bool holds only bools, so this reaches the expansion through a
    # prediction whose key set does not line up and is emptied wholesale.
    metric = _fed(make_entry, ({"verdicts": {"r1": True}}, {"verdicts": {"other": True}}))

    assert _flat(metric)["verdicts / r1"]["TF"] == 1


# --------------------------------------------------------------------------
# dict-of-bool expansion
# --------------------------------------------------------------------------


def test_a_dict_of_bools_becomes_one_sub_metric_per_sub_key(make_entry):
    metric = _fed(
        make_entry,
        ({"verdicts": {"r1": True, "r2": False}}, {"verdicts": {"r1": True, "r2": False}}),
    )

    results = _flat(metric)

    assert results["verdicts / r1"]["TT"] == 1
    assert results["verdicts / r2"]["FF"] == 1
    assert "verdicts" not in results


def test_a_sub_key_the_chain_omitted_is_a_mismatch(make_entry):
    metric = _fed(make_entry, ({"verdicts": {"r1": True}}, {"verdicts": {}}))

    assert _flat(metric)["verdicts / r1"]["TF"] == 1


def test_omitting_one_sub_key_leaves_the_others_scored_on_their_own_merit(make_entry):
    # The expansion exists so the report names the rule the chain struggles
    # with, so an omitted 'r2' must not be reported as 'r1' failing too. The
    # entry is still short of perfect, so this cannot disagree with matches().
    metric = _fed(
        make_entry,
        ({"verdicts": {"r1": True, "r2": True}}, {"verdicts": {"r1": True}}),
    )

    assert _flat(metric)["verdicts / r1"]["TT"] == 1
    assert _flat(metric)["verdicts / r2"]["TF"] == 1
    assert _flat(metric)["accuracy"] == pytest.approx(0.5)


def test_a_prediction_that_is_not_a_dict_scores_every_sub_key_as_missing(make_entry):
    # A string where a mapping was expected fails every sub-rule rather than
    # raising mid-run.
    metric = _fed(make_entry, ({"verdicts": {"r1": True, "r2": True}}, {"verdicts": "nope"}))

    assert _flat(metric)["verdicts / r1"]["TF"] == 1
    assert _flat(metric)["verdicts / r2"]["TF"] == 1


def test_an_unexpected_sub_key_fails_the_whole_rule_rather_than_scoring_a_perfect_one(
    make_entry,
):
    # values_agree needs the exact key set, so matches() fails this entry.
    # Crediting 'r1' reports 1.0 for an entry the grader fails.
    metric = _fed(
        make_entry,
        ({"verdicts": {"r1": True}}, {"verdicts": {"r1": True, "r2": True}}),
    )

    assert _flat(metric)["verdicts / r1"]["TF"] == 1
    assert _flat(metric)["accuracy"] == 0.0


def test_the_same_dict_key_expanding_on_every_entry_reuses_one_sub_metric(make_entry):
    # A path is the identity, so the same key expanding again is the next
    # entry rather than a second rule.
    metric = _fed(
        make_entry,
        ({"verdicts": {"r1": True}}, {"verdicts": {"r1": True}}),
        ({"verdicts": {"r1": True}}, {"verdicts": {"r1": True}}),
        ({"verdicts": {"r1": True}}, {"verdicts": {"r1": True}}),
    )

    assert _flat(metric)["verdicts / r1"]["TT"] == 3


def test_sub_keys_accumulate_across_entries_like_top_level_keys_do(make_entry):
    metric = _fed(
        make_entry,
        ({"verdicts": {"r1": True}}, {"verdicts": {"r1": True}}),
        ({"verdicts": {"r1": True}}, {"verdicts": {"r1": False}}),
    )

    assert _flat(metric)["verdicts / r1"]["accuracy"] == 0.5


# --------------------------------------------------------------------------
# Values that are neither bool nor number
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expected", "predicted", "agrees"),
    [
        ({"tag": "en"}, {"tag": "en"}, True),
        ({"tag": "en"}, {"tag": "de"}, False),
        ({"tags": [1, 2]}, {"tags": [1, 2]}, True),
        ({"tags": [1, 2]}, {"tags": [2, 1]}, False),
        ({"nothing": None}, {"nothing": None}, True),
        ({"v": {"value": "ok", "note": "why"}}, {"v": {"value": "ok", "note": "why"}}, True),
        ({"v": {"value": "ok", "note": "why"}}, {"v": {"value": "no", "note": "why"}}, False),
    ],
    ids=["str-same", "str-diff", "list-same", "list-diff", "none", "dict-same", "dict-diff"],
)
def test_a_value_that_is_neither_bool_nor_number_is_scored_by_agreement(
    make_entry, expected, predicted, agrees
):
    # Most corpora check such a key: echo checks a string, the tonality
    # templates the {value:, note:} shape.
    metric = _fed(make_entry, (expected, predicted))

    key = next(iter(expected))
    assert _flat(metric)[key]["accuracy"] == (1.0 if agrees else 0.0)
    assert _flat(metric)["accuracy"] == (1.0 if agrees else 0.0)


def test_a_value_rule_reports_matched_and_mismatched_counts(make_entry):
    metric = _fed(
        make_entry,
        ({"echo": "Hello there"}, {"echo": "Hello there"}),
        ({"echo": "Testing 123!"}, {"echo": "something else"}),
    )

    assert _flat(metric)["echo"] == {
        "accuracy": 0.5,
        "matched": 1,
        "mismatched": 1,
        "total": 2,
    }


def test_a_dict_value_that_is_not_all_bools_is_scored_whole(make_entry):
    # Only a dict of *bools* is expanded into per-sub-key metrics; anything
    # else is one value, compared as a whole by the same rules matches() uses.
    metric = _fed(
        make_entry,
        ({"verdicts": {"r1": 0.5}}, {"verdicts": {"r1": 0.5}}),
    )

    assert _flat(metric)["verdicts"]["matched"] == 1
    assert "verdicts / r1" not in _flat(metric)


# --------------------------------------------------------------------------
# The type guard runs on every entry, not just the first
# --------------------------------------------------------------------------


def test_a_bool_on_a_later_entry_is_not_arithmetic_in_a_numeric_metric(make_entry):
    # Mirror case: case 1 makes `score` numeric, so case 2's True must not be
    # subtracted as the integer 1.
    metric = SingleValueMetrics("Accuracy-Metrics")
    metric.add_entry(make_entry("a", expected={"score": 3}, predicted={"score": 3}))

    with pytest.raises(SystemExit, match="changes type between cases"):
        metric.add_entry(make_entry("b", expected={"score": True}, predicted={"score": 3}))


def test_a_key_that_changes_type_is_rejected_whichever_case_comes_first(make_entry):
    # Otherwise a late `passed: "maybe"` is graded by the metric case 1's
    # `passed: true` picked, and the same two cases swapped are the difference
    # between a clean run and a dead one.
    inconsistent = [
        ("a", {"passed": True}, {"passed": True}),
        ("b", {"passed": "maybe"}, {"passed": True}),
    ]

    for ordering in (inconsistent, list(reversed(inconsistent))):
        metric = SingleValueMetrics("Accuracy-Metrics")
        with pytest.raises(SystemExit, match="changes type between cases"):
            for entry_id, expected, predicted in ordering:
                metric.add_entry(make_entry(entry_id, expected=expected, predicted=predicted))


def test_an_empty_mapping_and_a_populated_one_are_different_kinds(make_entry):
    # Deliberate, not a gap: `{}` asserts "no keys here" as one whole
    # comparison, while `{r1: true}` expands into one sub-metric per rule.
    # The two cannot share a metric, so the corpus is rejected either way round.
    cases = [
        ("a", {"verdicts": {}}),
        ("b", {"verdicts": {"r1": True}}),
    ]

    for ordering in (cases, list(reversed(cases))):
        metric = SingleValueMetrics("Accuracy-Metrics")
        with pytest.raises(SystemExit, match="changes type between cases"):
            for entry_id, expected in ordering:
                metric.add_entry(make_entry(entry_id, expected=expected, predicted=expected))


def test_a_corpus_whose_mapping_is_empty_throughout_is_scored_whole(make_entry):
    # Consistent, so it runs: `verdicts: {}` is a real assertion - the
    # prediction has to be empty too - and ValueMetrics scores it as one key.
    metric = _fed(
        make_entry,
        ({"verdicts": {}}, {"verdicts": {}}),
        ({"verdicts": {}}, {"verdicts": {"r1": True}}),
    )

    assert _flat(metric)["verdicts"]["matched"] == 1
    assert _flat(metric)["verdicts"]["mismatched"] == 1
    assert "verdicts / r1" not in _flat(metric)


def test_the_type_change_message_names_the_values_not_the_internal_kinds(make_entry):
    # 'first seen as value, now bools' reads as nonsense next to two mappings.
    metric = SingleValueMetrics("Accuracy-Metrics")
    metric.add_entry(make_entry("a", expected={"verdicts": {}}, predicted={"verdicts": {}}))

    with pytest.raises(SystemExit) as raised:
        metric.add_entry(
            make_entry("b", expected={"verdicts": {"r1": True}}, predicted={"verdicts": {}})
        )

    assert "Case 'a': {}" in str(raised.value)
    assert "Case 'b': {'r1': True}" in str(raised.value)


def test_the_type_change_message_names_both_entries_not_just_the_later_one(make_entry):
    # A locator for whoever edits the .yaml: the entry that established the
    # kind is the half you cannot work out from the entry that broke it.
    metric = SingleValueMetrics("Accuracy-Metrics")
    metric.add_entry(make_entry("first-case", expected={"score": 0.9}, predicted={"score": 0.9}))

    with pytest.raises(SystemExit) as raised:
        metric.add_entry(
            make_entry("later-case", expected={"score": "high"}, predicted={"score": "high"})
        )

    assert "'first-case'" in str(raised.value)
    assert "'later-case'" in str(raised.value)


def test_int_and_float_on_the_same_key_are_the_same_kind(make_entry):
    # The guard must not reject a corpus that mixes 3 and 3.5 - both are
    # numbers and both belong in the same metric.
    metric = _fed(make_entry, ({"score": 3}, {"score": 3}), ({"score": 3.5}, {"score": 3.5}))

    assert _flat(metric)["score"]["scored"] == 2


# --------------------------------------------------------------------------
# The headline accuracy number
# --------------------------------------------------------------------------


def test_the_headline_accuracy_is_the_mean_across_boolean_rules(make_entry):
    metric = _fed(
        make_entry,
        ({"a": True, "b": True}, {"a": True, "b": False}),
    )

    # a scores 1.0, b scores 0.0
    assert _flat(metric)["accuracy"] == pytest.approx(0.5)


def test_numeric_rules_are_excluded_from_the_headline_accuracy(make_entry):
    # 'mae' is not on a fraction-correct scale, so it is tracked by its own
    # MAE chart instead of averaged into the headline - a wrong 'score' must
    # not drag down an otherwise-perfect 'passed'.
    metric = _fed(make_entry, ({"passed": True, "score": 1.0}, {"passed": True, "score": 0.0}))

    assert _flat(metric)["accuracy"] == 1.0


def test_a_purely_numeric_corpus_reports_a_headline_of_zero(make_entry):
    # A numeric-only corpus has no boolean/categorical rule for the headline
    # to average, so it reports 0.0 rather than folding 'mae' rules back in -
    # its real signal is the MAE chart, not this number.
    metric = _fed(
        make_entry,
        ({"room_count": 2}, {"room_count": 2}),
        ({"room_count": 1}, {"room_count": 5}),
    )

    assert _flat(metric)["accuracy"] == 0.0


def test_every_kind_of_rule_reports_an_accuracy_for_the_headline_to_average(make_entry):
    # What _calculate_mean_accuracy relies on rather than guarding for: a bool,
    # a number, a plain value and an expanded sub-rule all report an 'accuracy'
    # on the same fraction-correct scale. A fourth kind that does not has to
    # decide how it is weighted there.
    metric = _fed(
        make_entry,
        (
            {"passed": True, "score": 1.0, "tag": "en", "verdicts": {"r1": True}},
            {"passed": True, "score": 1.0, "tag": "en", "verdicts": {"r1": True}},
        ),
    )

    results = _flat(metric)

    assert {"passed", "score", "tag", "verdicts / r1"} <= set(results)
    for key in ("passed", "score", "tag", "verdicts / r1"):
        assert "accuracy" in results[key]
    assert results["accuracy"] == 1.0


# --------------------------------------------------------------------------
# The headline mean is weighted by how many entries each rule was checked on
# --------------------------------------------------------------------------


def test_a_rule_checked_on_fewer_cases_carries_proportionally_less_weight(
    make_entry,
):
    # Unweighted, a key present in 2 of 9 cases counts as much as one in all 9.
    metric = SingleValueMetrics("Accuracy-Metrics")
    # `passed` checked three times, all correct.
    for i in range(3):
        metric.add_entry(make_entry(f"p{i}", expected={"passed": True}, predicted={"passed": True}))
    # `extra` checked once, wrong.
    metric.add_entry(make_entry("e", expected={"extra": True}, predicted={"extra": False}))

    # 3 of 4 checks agreed, not the unweighted mean of 1.0 and 0.0.
    assert _flat(metric)["accuracy"] == pytest.approx(0.75)


def test_expanded_sub_keys_count_individually_toward_the_headline(make_entry):
    # Three rules, one of them wrong - not "the verdicts key is 50% right".
    metric = _fed(
        make_entry,
        (
            {"verdicts": {"r1": True, "r2": True}, "passed": True},
            {"verdicts": {"r1": True, "r2": False}, "passed": True},
        ),
    )

    assert _flat(metric)["accuracy"] == pytest.approx(2 / 3)


def test_an_unfed_metric_reports_no_rules_and_a_zero_accuracy():
    # A run with no rules at all divides by zero otherwise.
    assert SingleValueMetrics("Accuracy-Metrics").compute() == {"rules": {}, "accuracy": 0.0}


def test_compute_does_not_consume_what_was_added(make_entry):
    metric = _fed(make_entry, ({"passed": True}, {"passed": True}))

    assert _flat(metric) == _flat(metric)


# --------------------------------------------------------------------------
# Any corpus key is legal, whatever it is called
# --------------------------------------------------------------------------


def test_a_rule_named_accuracy_is_reported_like_any_other(make_entry):
    # Rules live under their own mapping, so a rule called 'accuracy' sits one
    # level below the headline of the same name and neither overwrites the
    # other.
    metric = _fed(
        make_entry,
        ({"accuracy": True}, {"accuracy": True}),
        ({"accuracy": True}, {"accuracy": False}),
    )

    assert _flat(metric)["accuracy"] == 0.5
    assert metric.compute()["rules"]["accuracy"]["results"]["TT"] == 1
    assert metric.compute()["accuracy"] == 0.5


@pytest.mark.parametrize(
    "name",
    ["a.b", "rules", "results", "sub_rules", "verdicts / r1", ""],
    ids=["dotted", "rules", "results", "sub-rules", "a-generated-label", "empty"],
)
def test_a_rule_may_be_named_after_any_structural_key(make_entry, name):
    # A corpus key is only ever a key *inside* 'rules', so none of the names
    # compute() uses structurally can be shadowed by one.
    metric = _fed(make_entry, ({name: True}, {name: True}))

    assert metric.compute()["rules"][name]["results"]["TT"] == 1
    assert metric.compute()["accuracy"] == 1.0


def test_a_dotted_key_and_the_sub_rule_it_looks_like_stay_separate(make_entry):
    # 'verdicts.r1' as a key of its own, beside the sub-rule a joined name
    # would call the same thing. Two rules, two metrics, each counted once.
    metric = _fed(
        make_entry,
        (
            {"verdicts": {"r1": True}, "verdicts.r1": False},
            {"verdicts": {"r1": True}, "verdicts.r1": True},
        ),
    )

    results = metric.compute()

    assert results["rules"]["verdicts"]["sub_rules"]["r1"]["results"]["TT"] == 1
    assert results["rules"]["verdicts.r1"]["results"]["FT"] == 1
    # Two rules, one right and one wrong.
    assert results["accuracy"] == 0.5


def test_two_dict_keys_whose_sub_rules_would_join_alike_stay_separate(make_entry):
    # 'a.b: {c}' and 'a: {b.c}' both join to 'a.b.c', which would share one
    # metric and double-weight it in the headline. As paths they are
    # ('a.b','c') and ('a','b.c') - different rules, and nothing to reject.
    metric = _fed(
        make_entry,
        (
            {"a.b": {"c": True}, "a": {"b.c": True}},
            {"a.b": {"c": True}, "a": {"b.c": False}},
        ),
    )

    results = metric.compute()

    assert results["rules"]["a.b"]["sub_rules"]["c"]["results"]["TT"] == 1
    assert results["rules"]["a"]["sub_rules"]["b.c"]["results"]["TF"] == 1
    assert results["accuracy"] == 0.5


# --------------------------------------------------------------------------
# The reported shape
# --------------------------------------------------------------------------


def test_a_top_level_rule_is_reported_under_results(make_entry):
    metric = _fed(make_entry, ({"passed": True}, {"passed": True}))

    assert set(metric.compute()) == {"rules", "accuracy"}
    assert set(metric.compute()["rules"]["passed"]) == {"results"}


def test_an_expanded_rule_is_reported_under_sub_rules(make_entry):
    metric = _fed(make_entry, ({"verdicts": {"r1": True}}, {"verdicts": {"r1": True}}))

    node = metric.compute()["rules"]["verdicts"]

    assert set(node) == {"sub_rules"}
    assert set(node["sub_rules"]["r1"]) == {"results"}


def test_iter_rule_results_yields_a_path_per_rule(make_entry):
    # The one traversal consumers use, so it has to reach both shapes.
    metric = _fed(
        make_entry,
        (
            {"passed": True, "verdicts": {"r1": True, "r2": False}},
            {"passed": True, "verdicts": {"r1": True, "r2": False}},
        ),
    )

    paths = [path for path, _ in iter_rule_results(metric.compute())]

    assert paths == [("passed",), ("verdicts", "r1"), ("verdicts", "r2")]


def test_the_traversal_helpers_are_reachable_from_the_package():
    # compute()'s shape is meant to be read through iter_rule_results rather
    # than walked by hand, so the package has to offer it - leaving it off sent
    # the one consumer doing it right past the package to the module.
    from chain_checker.baseclasses import metrics

    assert metrics.iter_rule_results is iter_rule_results
    assert metrics.rule_label is rule_label
    assert {"iter_rule_results", "rule_label", "RulePath"} <= set(metrics.__all__)


def test_rule_label_reads_a_path_as_text():
    assert rule_label(("passed",)) == "passed"
    assert rule_label(("verdicts", "r1")) == "verdicts / r1"
