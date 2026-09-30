"""Unit tests for chain_checker/baseclasses/metrics/m_value.py.

ValueMetrics is the fallback for an output key that is neither a bool nor a
number - a string, a list, or a mapping that is not a dict-of-bool. With no
confusion matrix or mae to report, it asks only what Label.matches() asks:
did the prediction agree? These tests pin that it asks on those exact terms,
so it and a graded PASS/FAIL cannot disagree about an entry.
"""

from chain_checker.baseclasses.metrics.m_value import ValueMetrics


def test_to_dict_reports_the_shape_every_metric_is_read_through():
    # The prompt modifier reads the three keys by name, and the name and
    # description are the ones this metric was built with.
    metric = ValueMetrics("Tonality", "d")
    metric.add("formal", "formal")

    assert metric.to_dict() == {
        "name": "Tonality",
        "description": "d",
        "results": metric.compute(),
    }


def test_an_agreeing_string_counts_as_matched():
    metric = ValueMetrics("Tonality", "d")
    metric.add("formal", "formal")

    assert metric.compute() == {
        "accuracy": 1.0,
        "matched": 1,
        "mismatched": 0,
        "total": 1,
    }


def test_a_disagreeing_string_counts_as_mismatched():
    metric = ValueMetrics("Tonality", "d")
    metric.add("formal", "casual")

    assert metric.compute() == {
        "accuracy": 0.0,
        "matched": 0,
        "mismatched": 1,
        "total": 1,
    }


def test_a_missing_prediction_cannot_agree():
    # What a parse failure and an omitted key both leave behind.
    metric = ValueMetrics("Tonality", "d")
    metric.add("formal", None)

    assert metric.compute()["matched"] == 0


def test_the_value_note_shape_the_tonality_templates_check_is_compared_whole():
    # The shape the tonality templates check.
    metric = ValueMetrics("Template", "d")
    metric.add(
        {"value": "formal", "note": "keep it short"}, {"value": "formal", "note": "keep it short"}
    )
    metric.add(
        {"value": "formal", "note": "keep it short"}, {"value": "formal", "note": "something else"}
    )

    assert metric.compute()["matched"] == 1
    assert metric.compute()["mismatched"] == 1


def test_a_nested_dict_carrying_an_extra_key_is_a_mismatch():
    # Ignoring unasked-for keys is a property of the top-level output block,
    # not of a value the corpus spelled out in full.
    metric = ValueMetrics("Template", "d")
    metric.add({"value": "formal"}, {"value": "formal", "note": "extra"})

    assert metric.compute()["matched"] == 0


def test_list_order_and_length_both_have_to_agree():
    metric = ValueMetrics("Keywords", "d")
    metric.add(["a", "b"], ["a", "b"])
    metric.add(["a", "b"], ["b", "a"])
    metric.add(["a", "b"], ["a"])

    assert metric.compute()["matched"] == 1
    assert metric.compute()["mismatched"] == 2


def test_a_string_where_a_bool_was_expected_is_a_mismatch_not_a_truthy_pass():
    # The bool/int split, reached through a string-valued corpus key.
    metric = ValueMetrics("Verdict", "d")
    metric.add("true", True)

    assert metric.compute()["matched"] == 0


def test_the_counts_always_add_up_to_the_total():
    metric = ValueMetrics("Tonality", "d")
    metric.add("formal", "formal")
    metric.add("formal", "casual")
    metric.add("formal", None)

    results = metric.compute()

    assert results["matched"] + results["mismatched"] == results["total"] == 3
    assert results["accuracy"] == 1 / 3


def test_an_unfed_metric_reports_zero_rather_than_a_vacuous_one():
    # `all([])` is True, so the empty case is decided deliberately: 1.0 would
    # make an unfed metric the best-scoring one in the report.
    assert ValueMetrics("Tonality", "d").compute() == {
        "accuracy": 0.0,
        "matched": 0,
        "mismatched": 0,
        "total": 0,
    }


def test_computing_twice_does_not_change_the_result():
    # The report renders it and the training loop serialises it separately.
    metric = ValueMetrics("Tonality", "d")
    metric.add("formal", "formal")

    assert metric.compute() == metric.compute()
