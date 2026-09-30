"""Unit tests for chain_checker/baseclasses/metrics/m_pair.py.

PairMetrics is the collector the three pairwise metrics share. It decides
nothing about correctness - it only keeps the two series aligned, which is
what lets every subclass zip them and divide by the length of one.
"""

import pytest

from chain_checker.baseclasses.metrics.m_boolean import BooleanMetrics
from chain_checker.baseclasses.metrics.m_float import FloatMetrics
from chain_checker.baseclasses.metrics.m_pair import PairMetrics
from chain_checker.baseclasses.metrics.m_value import ValueMetrics


class _Pairs(PairMetrics):
    """The smallest subclass: reports the pairs it holds."""

    def compute(self) -> list[tuple]:
        return list(zip(self._true_values, self._pred_values, strict=True))


def test_an_unfed_collector_holds_no_pairs():
    assert _Pairs("Pairs", "d").compute() == []


def test_pairs_are_kept_in_the_order_they_were_added():
    metric = _Pairs("Pairs", "d")
    metric.add(1, 2)
    metric.add(3, 4)

    assert metric.compute() == [(1, 2), (3, 4)]


def test_a_missing_value_on_either_side_still_takes_a_slot():
    # The two series are zipped and one of them is the denominator, so a value
    # skipped rather than recorded would shift the pairing for every entry
    # after it.
    metric = _Pairs("Pairs", "d")
    metric.add(1, None)
    metric.add(None, 2)

    assert metric.compute() == [(1, None), (None, 2)]


def test_the_pairwise_metrics_all_collect_through_it():
    # One collector for the three, so their add() signatures cannot drift
    # apart and compute() can rely on the two series being aligned.
    for metric_class in (BooleanMetrics, FloatMetrics, ValueMetrics):
        assert issubclass(metric_class, PairMetrics)
        assert "add" not in vars(metric_class)


def test_the_two_metrics_reporting_agreement_get_it_from_one_place():
    # FloatMetrics and ValueMetrics both report 'accuracy' as the fraction of
    # pairs that agreed, on the same 0..1 scale the report averages them on. One
    # owner, so a change to what counts as agreement moves both together.
    pairs = [(1.0, 1.0), (2.0, 9.9), (3.0, None), (0.3, 0.1 + 0.2)]

    numeric, value = FloatMetrics("N", "d"), ValueMetrics("V", "d")
    for true_value, pred_value in pairs:
        numeric.add(true_value, pred_value)
        value.add(true_value, pred_value)

    assert numeric.compute()["accuracy"] == value.compute()["accuracy"] == 0.5


def test_an_unfed_pair_metric_agrees_on_nothing_rather_than_vacuously():
    # `all([])` is True, so the empty case is decided once here rather than
    # separately in each subclass.
    assert _Pairs("Pairs", "d")._agreement() == (0, 0)


def test_the_collector_it_does_not_implement_still_refuses():
    # PairMetrics answers add() only; the base class's other two refusals have
    # to survive the subclassing.
    metric = _Pairs("Pairs", "d")

    with pytest.raises(NotImplementedError, match="add_item"):
        metric.add_item("anything")
