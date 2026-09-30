"""Unit tests for chain_checker/baseclasses/metrics/m_list.py.

ListMetrics is the other descriptive metric: a series rather than a tally, so
duplicates are the point. It scores nothing either - it shows the modifier LLM
the shape of the corpus it is writing a prompt for. Fed through add_item().
"""

from chain_checker.baseclasses.metrics.m_list import ListMetrics


def test_an_unfed_list_metric_reports_an_empty_list():
    assert ListMetrics("Modification", "d").compute() == []


def test_items_are_reported_in_the_order_they_were_added():
    metric = ListMetrics("Modification", "d")
    for modification in ("swap", "typo", "swap"):
        metric.add_item(modification)

    assert metric.compute() == ["swap", "typo", "swap"]


def test_duplicates_are_kept_rather_than_collapsed():
    # The difference from KeywordMetrics: this one is a series, not a tally,
    # and ModifierMetrics uses it for word counts where repeats are the point.
    metric = ListMetrics("TextLength", "d")
    for length in (12, 12, 45):
        metric.add_item(length)

    assert metric.compute() == [12, 12, 45]


def test_an_unhashable_element_is_kept_like_any_other():
    # ListMetrics appends rather than counting, which is why ModifierMetrics
    # feeds this one without the hashability guard the two tallies need.
    metric = ListMetrics("Modification", "d")
    metric.add_item(["swap", "typo"])

    assert metric.compute() == [["swap", "typo"]]


def test_the_reported_list_is_a_copy():
    # The results are mutated and serialised downstream, so compute() must not
    # hand out the container it accumulates into.
    metric = ListMetrics("TextLength", "d")
    metric.add_item(12)

    metric.compute().append(99)

    assert metric.compute() == [12]


def test_list_items_survive_being_computed_twice():
    metric = ListMetrics("TextLength", "d")
    metric.add_item(12)

    assert metric.compute() == metric.compute() == [12]
