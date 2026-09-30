"""Unit tests for chain_checker/baseclasses/metrics/m_tokens.py.

TokenUsageMetrics counts what the chain under test cost, not what the
modifier LLM cost. A recalled entry arrives carrying the original call's
usage, because the recall path parses the saved TOKEN USAGE
block back into the ModelOutput. The last test pins the description against
that, since it reaches the modifier LLM verbatim.
"""

import pytest

from chain_checker.baseclasses.metrics.m_tokens import TokenUsageMetrics


def _usage(prompt=0, completion=0, total=0) -> dict:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def test_a_run_with_no_usage_at_all_reports_zero_instead_of_dividing_by_zero():
    assert TokenUsageMetrics().compute() == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "entries_with_usage": 0,
        "avg_tokens_per_entry": 0.0,
    }


def test_usage_is_summed_across_the_entries_that_made_a_call(make_entry):
    metric = TokenUsageMetrics()
    metric.add_entry(make_entry("a", token_usage=_usage(10, 5, 15)))
    metric.add_entry(make_entry("b", token_usage=_usage(20, 10, 30)))

    assert metric.compute() == {
        "prompt_tokens": 30,
        "completion_tokens": 15,
        "total_tokens": 45,
        "entries_with_usage": 2,
        "avg_tokens_per_entry": pytest.approx(22.5),
    }


def test_the_average_divides_by_calls_made_not_by_corpus_size(make_entry):
    # Three entries, one call. An entry with no usage is skipped rather than
    # counted as a zero-token call, which would drag the average down and claim
    # a call was made that never was.
    metric = TokenUsageMetrics()
    for entry_id in ("a", "b"):
        metric.add_entry(make_entry(entry_id))
    metric.add_entry(make_entry("c", token_usage=_usage(60, 40, 100)))

    results = metric.compute()

    assert results["entries_with_usage"] == 1
    assert results["avg_tokens_per_entry"] == pytest.approx(100.0)


def test_a_usage_dict_missing_a_field_contributes_zero_for_it(make_entry):
    # Different backends report different fields (see the modifier backends) - a
    # missing one must not take the run down mid-report.
    metric = TokenUsageMetrics()
    metric.add_entry(make_entry("a", token_usage={"total_tokens": 15}))

    results = metric.compute()

    assert results["prompt_tokens"] == 0
    assert results["total_tokens"] == 15


def test_a_usage_dict_without_a_total_is_summed_from_its_two_parts(make_entry):
    # Every producer of a usage dict falls back to prompt + completion when the
    # backend reports no total. Without it the epoch aggregates as costing
    # zero, which the summary's cost tie-breaker reads as the cheapest prompt.
    metric = TokenUsageMetrics()
    metric.add_entry(make_entry("a", token_usage={"prompt_tokens": 1000, "completion_tokens": 200}))

    results = metric.compute()

    assert results["total_tokens"] == 1200
    assert results["avg_tokens_per_entry"] == pytest.approx(1200.0)


def test_a_reported_total_wins_over_the_sum_of_the_parts(make_entry):
    # The fallback must not start recomputing a total the backend gave us -
    # some report a total that includes tokens neither part covers.
    metric = TokenUsageMetrics()
    metric.add_entry(make_entry("a", token_usage=_usage(10, 5, 99)))

    assert metric.compute()["total_tokens"] == 99


def test_entries_with_and_without_a_total_sum_together(make_entry):
    metric = TokenUsageMetrics()
    metric.add_entry(make_entry("a", token_usage=_usage(10, 5, 15)))
    metric.add_entry(make_entry("b", token_usage={"prompt_tokens": 20, "completion_tokens": 10}))

    assert metric.compute()["total_tokens"] == 45


def test_an_all_zero_usage_dict_is_not_counted_as_a_call(make_entry):
    # {"total_tokens": 0} is a truthy dict but carries no measurable usage -
    # gated on the same total compute() derives, not on dict non-emptiness, so
    # this must not inflate 'entries_with_usage' and drag the average down.
    metric = TokenUsageMetrics()
    metric.add_entry(make_entry("a", token_usage=_usage()))

    assert metric.compute()["entries_with_usage"] == 0
    assert metric.compute()["avg_tokens_per_entry"] == 0.0


def test_an_empty_usage_dict_is_treated_as_no_call_at_all(make_entry):
    # EmptyModelOutput and any recalled entry without usage both give {},
    # whose total is zero either way.
    metric = TokenUsageMetrics()
    metric.add_entry(make_entry("a", token_usage={}))

    assert metric.compute()["entries_with_usage"] == 0


def test_the_totals_and_the_two_parts_read_a_field_the_same_way(make_entry):
    # A recalled entry's usage comes back through json.loads, so a hand-edited
    # cache file can hand every field over as a str. One reading, so the
    # aggregate total cannot survive a value the per-part sums choke on.
    metric = TokenUsageMetrics()
    metric.add_entry(make_entry("a", token_usage={"prompt_tokens": "10", "completion_tokens": "5"}))

    assert metric.compute() == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "entries_with_usage": 1,
        "avg_tokens_per_entry": pytest.approx(15.0),
    }


def test_a_null_usage_field_counts_as_zero_rather_than_raising(make_entry):
    # A backend reporting a field it did not measure spells it null, which
    # int(None) refuses.
    metric = TokenUsageMetrics()
    metric.add_entry(make_entry("a", token_usage={"total_tokens": None, "prompt_tokens": 7}))

    assert metric.compute()["total_tokens"] == 7


def test_a_usage_field_that_is_no_number_at_all_counts_as_zero(make_entry):
    # The same hand-edited cache file that types "10" as a str can carry
    # "unknown" or a container. compute() runs at the end of an epoch that has
    # already been paid for, so one unreadable field must not take it down.
    metric = TokenUsageMetrics()
    metric.add_entry(
        make_entry("a", token_usage={"prompt_tokens": "unknown", "completion_tokens": 5})
    )
    # b's only field is unreadable, so its usage totals zero - the same as no
    # usage at all, and it is excluded from 'entries_with_usage' accordingly.
    metric.add_entry(make_entry("b", token_usage={"total_tokens": ["not", "a", "count"]}))

    results = metric.compute()

    assert results["prompt_tokens"] == 0
    assert results["completion_tokens"] == 5
    assert results["total_tokens"] == 5
    assert results["entries_with_usage"] == 1


def test_an_infinite_usage_field_counts_as_zero_rather_than_raising(make_entry):
    # json.loads accepts "Infinity" by default, so a hand-edited cache file
    # can hand back a bare float('inf') - and int(float('inf')) raises
    # OverflowError, which used to escape uncaught.
    metric = TokenUsageMetrics()
    metric.add_entry(make_entry("a", token_usage={"total_tokens": float("inf")}))

    assert metric.compute()["total_tokens"] == 0


def test_a_bool_usage_field_counts_as_zero_not_as_one(make_entry):
    # bool is an int subclass, so int(True) == 1 - a mistyped field must not
    # silently score a token count that was never reported. No prompt/
    # completion fallback here, so a wrongly-counted 1 would surface directly.
    metric = TokenUsageMetrics()
    metric.add_entry(make_entry("a", token_usage={"total_tokens": True}))

    assert metric.compute()["total_tokens"] == 0


def test_a_bool_total_falls_back_to_the_real_reported_parts(make_entry):
    # total_tokens=True must not report 1 - excluded like any other unreadable
    # field, so the real prompt/completion parts win instead.
    metric = TokenUsageMetrics()
    metric.add_entry(make_entry("a", token_usage={"total_tokens": True, "prompt_tokens": 5}))

    results = metric.compute()

    assert results["total_tokens"] == 5
    assert results["entries_with_usage"] == 1


def test_a_non_integer_float_usage_field_counts_as_zero_not_truncated(make_entry):
    # int(3.9) == 3 silently truncates - a fractional token count is not a
    # real backend value, so it is treated as unreadable rather than rounded.
    metric = TokenUsageMetrics()
    metric.add_entry(make_entry("a", token_usage={"total_tokens": 3.9}))

    assert metric.compute()["total_tokens"] == 0


def test_compute_does_not_consume_what_was_added(make_entry):
    metric = TokenUsageMetrics()
    metric.add_entry(make_entry("a", token_usage=_usage(10, 5, 15)))

    assert metric.compute() == metric.compute()


def test_a_recalled_entry_contributes_and_the_description_says_so(make_entry):
    # A recalled entry arrives with the original call's real usage, so its
    # total is non-zero and add_entry keeps it. The description has to say
    # so: it reaches the modifier LLM verbatim, which weighs accuracy against
    # cost on it.
    recalled = make_entry("recalled-from-disk", token_usage=_usage(10, 5, 15))

    metric = TokenUsageMetrics()
    metric.add_entry(recalled)

    assert metric.compute()["entries_with_usage"] == 1
    assert metric.compute()["total_tokens"] == 15

    description = TokenUsageMetrics().get_description()
    assert "keeps the usage" in description
