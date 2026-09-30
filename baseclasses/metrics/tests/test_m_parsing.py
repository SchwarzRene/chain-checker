"""Unit tests for chain_checker/baseclasses/metrics/m_parsing.py.

ParsingMetrics separates "the model answered wrongly" from "the model never
produced a usable answer". It is the one metric fed by set() rather than
add_entry(), because which entries failed to parse is something the loop
knows and an Entry does not.
"""

from chain_checker.baseclasses.metrics.m_parsing import ParsingMetrics


def test_an_unset_metric_reports_a_zeroed_run_rather_than_dividing_by_zero():
    # ModifierMetrics always calls set(), but the report renders a metric
    # that was built and never fed - an empty corpus reaches exactly this.
    assert ParsingMetrics().compute() == {
        "total": 0,
        "parsed": 0,
        "failed": 0,
        "failure_rate": 0.0,
        "failed_ids": [],
    }


def test_a_clean_run_reports_every_entry_parsed():
    metric = ParsingMetrics()
    metric.set(10, set())

    assert metric.compute() == {
        "total": 10,
        "parsed": 10,
        "failed": 0,
        "failure_rate": 0.0,
        "failed_ids": [],
    }


def test_failures_are_subtracted_from_the_parsed_count():
    metric = ParsingMetrics()
    metric.set(10, {"a", "b"})

    results = metric.compute()

    assert results["parsed"] == 8
    assert results["failed"] == 2
    assert results["failure_rate"] == 0.2


def test_a_totally_unparseable_run_reports_a_failure_rate_of_one():
    metric = ParsingMetrics()
    metric.set(3, {1, 2, 3})

    assert metric.compute()["failure_rate"] == 1.0
    assert metric.compute()["parsed"] == 0


def test_failed_ids_are_stored_and_sorted_as_text_so_mixed_id_types_do_not_raise():
    # Corpus ids can be ints or strings in the same run, and sorted() on a
    # mixed set raises TypeError - so ids are coerced to str before sorting,
    # not just sorted by a str key, keeping this list str-only like
    # Negative-Predicted-Metrics' entry ids.
    metric = ParsingMetrics()
    metric.set(5, {10, "b", 2, "a"})

    assert metric.compute()["failed_ids"] == ["10", "2", "a", "b"]


def test_set_replaces_rather_than_accumulates():
    # The training loop reuses a metric object across a re-run of the same epoch.
    metric = ParsingMetrics()
    metric.set(10, {"a", "b"})
    metric.set(4, {"c"})

    assert metric.compute()["total"] == 4
    assert metric.compute()["failed_ids"] == ["c"]


def test_a_list_of_failed_ids_is_accepted_as_well_as_a_set():
    # set() only needs an iterable, and the sorted() copy means the caller's
    # collection is not held by reference.
    failed = ["b", "a"]
    metric = ParsingMetrics()
    metric.set(3, failed)

    failed.append("c")

    assert metric.compute()["failed_ids"] == ["a", "b"]


def test_the_reported_failed_ids_are_a_copy():
    # The way out, as the test above covers the way in. The live list lets a
    # consumer change 'failed'/'parsed'/'failure_rate' on the next compute().
    metric = ParsingMetrics()
    metric.set(10, {"a", "b"})

    results = metric.compute()
    results["failed_ids"].append("not-a-real-failure")

    assert metric.compute()["failed_ids"] == ["a", "b"]
    assert metric.compute()["failed"] == 2
    assert metric.compute()["parsed"] == 8
    assert metric.compute()["failure_rate"] == 0.2


def test_each_compute_hands_out_a_distinct_list():
    # Two consumers holding the same run's results must not share a list
    # either - the report renders one while the training loop serialises the other.
    metric = ParsingMetrics()
    metric.set(3, {"a"})

    assert metric.compute()["failed_ids"] is not metric.compute()["failed_ids"]


def test_a_repeated_failed_id_is_counted_once():
    # 'failed' is len(failed_ids), so an undeduplicated repeat reports more
    # failures than the corpus has entries. The loop itself passes a set.
    metric = ParsingMetrics()
    metric.set(2, ["a", "a"])

    assert metric.compute()["failed"] == 1
    assert metric.compute()["parsed"] == 1
    assert metric.compute()["failed_ids"] == ["a"]


def test_more_failures_than_entries_reports_a_negative_parsed_count():
    # The limit the ponytail in set() names: nothing reconciles these ids
    # against the corpus, so an id that is not in it drives 'parsed' below zero
    # and 'failure_rate' above 1.0 - both of which the report renders as-is.
    metric = ParsingMetrics()
    metric.set(1, {"a", "not-in-the-corpus"})

    assert metric.compute()["parsed"] == -1
    assert metric.compute()["failure_rate"] == 2.0
