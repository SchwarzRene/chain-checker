"""Unit tests for chain_checker/baseclasses/metrics/m_boolean.py.

SingleValueMetrics routes every boolean output key through this class, so it
is where the run's headline accuracy ultimately comes from. The part worth
pinning down here is the quadrant orientation: TF and FT read easily as each
other, and a swap would leave 'accuracy' untouched while flipping what the
report claims the chain got wrong.
"""

from chain_checker.baseclasses.metrics.m_boolean import BooleanMetrics


def _metric(*pairs) -> BooleanMetrics:
    metric = BooleanMetrics("Bool", "d")
    for true_value, pred_value in pairs:
        metric.add(true_value, pred_value)
    return metric


def test_an_empty_metric_reports_zero_accuracy_rather_than_dividing_by_zero():
    # A corpus where not one entry carries this key still has to produce a
    # renderable result rather than a ZeroDivisionError.
    assert _metric().compute() == {
        "accuracy": 0.0,
        "TT": 0,
        "TF": 0,
        "FT": 0,
        "FF": 0,
    }


def test_each_quadrant_counts_the_pair_its_name_describes():
    # Read as (expected, predicted): TF means "expected true, predicted
    # false", never the other way around.
    assert _metric((True, True)).compute()["TT"] == 1
    assert _metric((True, False)).compute()["TF"] == 1
    assert _metric((False, True)).compute()["FT"] == 1
    assert _metric((False, False)).compute()["FF"] == 1


def test_accuracy_is_the_agreeing_diagonal_over_the_total():
    metric = _metric((True, True), (False, False), (True, False), (False, True))

    assert metric.compute() == {
        "accuracy": 0.5,
        "TT": 1,
        "TF": 1,
        "FT": 1,
        "FF": 1,
    }


def test_a_perfect_and_a_hopeless_run_are_the_two_ends_of_the_scale():
    assert _metric((True, True), (False, False)).compute()["accuracy"] == 1.0
    assert _metric((True, False), (False, True)).compute()["accuracy"] == 0.0


def test_compute_does_not_consume_what_was_added():
    # The same metric instance gets read more than once - by the epoch
    # report, the modifier prompt, and again when the training loop writes
    # its summary - so compute() cannot drain what it holds.
    metric = _metric((True, True), (True, False))

    assert metric.compute() == metric.compute()


def test_adding_after_a_compute_is_reflected_in_the_next_one():
    metric = _metric((True, True))
    assert metric.compute()["accuracy"] == 1.0

    metric.add(True, False)

    assert metric.compute()["accuracy"] == 0.5


# --------------------------------------------------------------------------
# Agreement is decided by label.values_agree, not by truthiness
# --------------------------------------------------------------------------


def test_a_non_bool_prediction_never_counts_as_agreement():
    # A plain truthiness check would read 1, "yes" and [0] as all agreeing
    # with an expected True, but Label.matches() rejects every one of them -
    # and since this metric grades through the same values_agree(), it
    # cannot land on the opposite verdict for the same entry.
    assert _metric((True, 1)).compute()["TF"] == 1
    assert _metric((True, "yes")).compute()["TF"] == 1
    assert _metric((True, [0])).compute()["TF"] == 1
    assert _metric((True, 1)).compute()["accuracy"] == 0.0


def test_a_mistyped_falsy_prediction_never_counts_as_agreement_either():
    # 0 and "" are falsy, so this is the direction a naive check would get
    # wrong: an expected False must not be satisfied by a value of the wrong
    # type just because it happens to be falsy too.
    assert _metric((False, 0)).compute()["FT"] == 1
    assert _metric((False, "")).compute()["FT"] == 1
    assert _metric((False, 0)).compute()["accuracy"] == 0.0


def test_a_missing_prediction_is_a_disagreement_whichever_value_was_expected():
    # A key the model never produced, or a whole entry that failed to parse,
    # both arrive here as None - and if that were ever read as agreeing with
    # an expected False, a run that answered nothing would score well.
    assert _metric((True, None)).compute()["TF"] == 1
    assert _metric((False, None)).compute()["FT"] == 1
    assert _metric((True, None), (False, None)).compute()["accuracy"] == 0.0


def test_the_quadrant_follows_the_expected_value_and_the_disagreement():
    # An identical, nonsensical prediction still has to sort into the
    # quadrant that names what was actually expected, not the one that
    # happens to match the garbage value itself.
    results = _metric((True, "garbage"), (False, "garbage")).compute()

    assert results["TF"] == 1
    assert results["FT"] == 1
    assert results["TT"] == results["FF"] == 0
