"""Unit tests for chain_checker/baseclasses/metrics/m_float.py.

FloatMetrics is the numeric counterpart to BooleanMetrics: it answers "how
far off", not "right or wrong", so it deliberately has no notion of a pass.
Nothing constructs one directly - SingleValueMetrics._metric_for routes a
numeric key here.
"""

import pytest

from chain_checker.baseclasses.metrics.m_float import FloatMetrics


def _metric(*pairs) -> FloatMetrics:
    metric = FloatMetrics("Score", "d")
    for true_value, pred_value in pairs:
        metric.add(true_value, pred_value)
    return metric


def test_an_empty_metric_reports_a_zero_mae_rather_than_dividing_by_zero():
    assert _metric().compute() == {
        "mae": 0.0,
        "accuracy": 0.0,
        "scored": 0,
        "unscored": 0,
        "true_scores": [],
        "predicted_scores": [],
    }


def test_mae_is_the_mean_absolute_error():
    metric = _metric((1.0, 1.5), (2.0, 1.0))

    assert metric.compute()["mae"] == pytest.approx(0.75)


def test_the_error_is_absolute_so_over_and_under_shooting_do_not_cancel():
    # Signed errors cancel, reporting a perfect 0.0 for a chain that is wildly
    # wrong in both directions.
    metric = _metric((1.0, 2.0), (1.0, 0.0))

    assert metric.compute()["mae"] == pytest.approx(1.0)


def test_the_raw_values_are_reported_in_the_order_they_were_added():
    # The report plots these as a series, so order is part of the contract.
    metric = _metric((0.1, 0.2), (0.3, 0.4))

    assert metric.compute()["true_scores"] == [0.1, 0.3]
    assert metric.compute()["predicted_scores"] == [0.2, 0.4]


def test_a_missing_prediction_is_left_out_of_the_mae_but_kept_in_the_series():
    # Treating None as 0.0 would report an error the chain never made, but the
    # None stays in the series so the gap is visible.
    metric = _metric((1.0, 1.0), (2.0, None))

    assert metric.compute()["mae"] == pytest.approx(0.0)
    assert metric.compute()["predicted_scores"] == [1.0, None]
    assert metric.compute()["true_scores"] == [1.0, 2.0]


def test_ints_are_accepted_alongside_floats():
    # _metric_for routes int and float to the same metric, so a corpus with
    # `count: 3` lands here.
    assert _metric((3, 5)).compute()["mae"] == pytest.approx(2.0)


def test_compute_does_not_consume_what_was_added():
    metric = _metric((1.0, 2.0))

    assert metric.compute() == metric.compute()


def test_a_null_expected_value_is_skipped_rather_than_crashing_the_report():
    # A corpus case can spell a checked number as null. Filtering the
    # prediction alone leaves it reaching abs(None - 20.0).
    metric = _metric((24.5, 20.0), (None, 20.0))

    results = metric.compute()

    assert results["mae"] == pytest.approx(4.5)
    assert results["scored"] == 1
    assert results["unscored"] == 1


def test_a_non_numeric_prediction_is_skipped_rather_than_crashing_the_report():
    # The same crash from the other side: a chain returning the string "0.5"
    # where the corpus expects 0.5.
    metric = _metric((1.0, 1.5), (1.0, "0.5"))

    results = metric.compute()

    assert results["mae"] == pytest.approx(0.5)
    assert results["scored"] == 1
    assert results["unscored"] == 1


def test_a_non_numeric_prediction_is_reported_as_a_gap_not_verbatim():
    # Skipping it in 'mae' is only half: emitted verbatim, the string reaches
    # the paired-bar chart's max() and raises on str vs float. Non-numerics
    # collapse to the None it draws as a dash, keeping the series aligned.
    metric = _metric((1.0, 1.5), (1.0, "0.5"), (1.0, {"nested": 1}))

    results = metric.compute()

    assert results["predicted_scores"] == [1.5, None, None]
    assert results["true_scores"] == [1.0, 1.0, 1.0]


def test_a_non_numeric_expected_value_is_reported_as_a_gap_too():
    # Both sides are filtered for 'mae', so both are coerced here.
    metric = _metric((None, 1.0), ("high", 2.0))

    assert metric.compute()["true_scores"] == [None, None]
    assert metric.compute()["predicted_scores"] == [1.0, 2.0]


def test_the_reported_series_carry_only_numbers_and_gaps():
    # What the report's paired-bar chart needs of them: it takes a max() over
    # the values, so a string or a mapping reaching it raises. Asserted on the
    # series rather than against the renderer, which lives in another package.
    results = _metric((1.0, 1.5), (1.0, "0.5"), (None, 2.0)).compute()

    for series in (results["true_scores"], results["predicted_scores"]):
        assert all(value is None or isinstance(value, (int, float)) for value in series)


def test_a_bool_is_not_treated_as_the_number_one():
    # bool is a subclass of int, so `abs(3 - True)` quietly computes 2. A True
    # where a number was expected is a type error, not the value 1 - the same
    # split values_agree makes.
    metric = _metric((3, True))

    assert metric.compute()["scored"] == 0
    assert metric.compute()["accuracy"] == 0.0


def test_the_reported_series_are_copies():
    # A consumer sorting or truncating the series it is handed must not rewrite
    # the metric.
    metric = _metric((1.0, 2.0))

    metric.compute()["true_scores"].append(99.0)

    assert metric.compute()["true_scores"] == [1.0]


# --------------------------------------------------------------------------
# accuracy - the same question Label.matches() asks
# --------------------------------------------------------------------------


def test_accuracy_counts_how_often_the_prediction_agreed_within_tolerance():
    # Without it a purely numeric corpus reports a flat 0.0 headline.
    # 0.1 + 0.2 != 0.3 exactly, so this also pins the isclose tolerance.
    metric = _metric((0.3, 0.1 + 0.2), (1.0, 2.0))

    assert metric.compute()["accuracy"] == pytest.approx(0.5)


def test_accuracy_covers_every_entry_not_just_the_scoreable_ones():
    # An all-missing run is indistinguishable in 'mae' from a chain that got
    # everything exactly right - 'accuracy', 'scored' and 'predicted_scores'
    # are what tell the two apart, since a missing prediction cannot agree.
    metric = _metric((1.0, None), (2.0, None))

    results = metric.compute()

    assert results["mae"] == 0.0
    assert results["scored"] == 0
    assert results["accuracy"] == 0.0
    assert results["predicted_scores"] == [None, None]


def test_a_mostly_missing_run_reports_how_little_the_mae_covers():
    # Nine of ten missing must not report the surviving entry's error as the
    # whole corpus's mae with nothing saying how much it covered.
    metric = _metric(*([(1.0, 1.0)] + [(1.0, None)] * 9))

    results = metric.compute()

    assert results["mae"] == 0.0
    assert results["scored"] == 1
    assert results["unscored"] == 9
    assert results["accuracy"] == pytest.approx(0.1)
