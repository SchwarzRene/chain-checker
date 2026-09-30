import pytest

from chain_checker.baseclasses.loop.center import ClassificationLoop
from chain_checker.baseclasses.metrics import ModifierMetrics

from .conftest import FakeModel


@pytest.fixture
def run(base_dir, make_corpus):

    def _run(model, *entry_ids, run_id="modification_0") -> dict:
        corpus = make_corpus(*entry_ids)
        loop = ClassificationLoop(run_id, corpus, model, base_dir)
        loop.loop()

        metrics = ModifierMetrics(corpus, failed_ids=loop.get_failed_ids())
        return {item["name"]: item["results"] for item in metrics.get_metrics()}

    return _run


def test_a_run_grades_the_corpus_it_just_walked(run):

    model = FakeModel(answers={"b": {"passed": False}})

    results = run(model, "a", "b", "c")

    assert results["Accuracy-Metrics"]["accuracy"] == pytest.approx(2 / 3)


def test_a_wrong_answer_reaches_the_counter_examples_by_id(run):

    results = run(FakeModel(answers={"b": {"passed": False}}), "a", "b", "c")

    assert set(results["Negative-Predicted-Metrics"]) == {"b"}


def test_every_entry_that_answered_counts_as_parsed(run):
    results = run(FakeModel(), "a", "b", "c")

    assert results["Parsing-Metrics"] == {
        "total": 3,
        "parsed": 3,
        "failed": 0,
        "failure_rate": 0.0,
        "failed_ids": [],
    }


def test_what_the_run_cost_is_summed_across_the_corpus(run):
    results = run(FakeModel(token_usage={"total_tokens": 5}), "a", "b", "c")

    usage = results["Chain-Token-Usage-Metrics"]
    assert usage["total_tokens"] == 15
    assert usage["entries_with_usage"] == 3


def test_an_entry_nothing_answered_is_reported_as_a_parsing_failure(run):
    results = run(FakeModel(fails=["b"]), "a", "b", "c")

    parsing = results["Parsing-Metrics"]
    assert parsing["failed_ids"] == ["b"]
    assert parsing["parsed"] == 2


def test_an_entry_nothing_answered_still_counts_as_wrong(run):

    results = run(FakeModel(fails=["b"]), "a", "b", "c")

    assert results["Accuracy-Metrics"]["accuracy"] == pytest.approx(2 / 3)


def test_an_entry_nothing_answered_is_shown_as_a_counter_example(run):
    results = run(FakeModel(fails=["b"]), "a", "b", "c")

    assert set(results["Negative-Predicted-Metrics"]) == {"b"}


def test_a_failed_entry_costs_nothing_and_is_not_counted_as_a_call(run):

    results = run(FakeModel(fails=["b"]), "a", "b", "c")

    assert results["Chain-Token-Usage-Metrics"]["entries_with_usage"] == 2


def test_a_run_where_nothing_answered_reports_a_full_failure_rate(run):
    results = run(FakeModel(fails=["a", "b"]), "a", "b")

    assert results["Parsing-Metrics"]["failure_rate"] == 1.0
    assert results["Accuracy-Metrics"]["accuracy"] == 0.0


def test_a_recalled_run_reports_exactly_what_the_paid_run_reported(run):

    first = run(FakeModel(answers={"b": {"passed": False}}), "a", "b", "c")

    second = run(FakeModel(answers={"b": {"passed": False}}), "a", "b", "c")

    assert second == first


def test_a_recalled_run_makes_no_calls_at_all(run):
    run(FakeModel(), "a", "b", "c")

    recalled = FakeModel()
    run(recalled, "a", "b", "c")

    assert recalled.inputs == []


def test_a_recalled_run_still_reports_what_the_corpus_cost_to_produce(run):

    run(FakeModel(token_usage={"total_tokens": 5}), "a", "b", "c")

    results = run(FakeModel(token_usage={"total_tokens": 5}), "a", "b", "c")

    assert results["Chain-Token-Usage-Metrics"]["total_tokens"] == 15


def test_an_entry_that_failed_is_the_only_one_a_re_run_calls(run):

    run(FakeModel(fails=["b"]), "a", "b", "c")

    retried = FakeModel()
    results = run(retried, "a", "b", "c")

    assert retried.inputs == ["b"]
    assert results["Parsing-Metrics"]["failed_ids"] == []


def test_a_rewritten_prompt_regrades_the_whole_corpus(run):

    run(FakeModel(), "a", "b", "c")

    rewritten = FakeModel(prompt="a better prompt", answers={"a": {"passed": False}})
    results = run(rewritten, "a", "b", "c")

    assert rewritten.inputs == ["a", "b", "c"]
    assert results["Accuracy-Metrics"]["accuracy"] == pytest.approx(2 / 3)


def test_two_runs_of_one_corpus_under_different_ids_do_not_share_a_cache(run):
    run(FakeModel(), "a", "b")

    other_epoch = FakeModel()
    run(other_epoch, "a", "b", run_id="modification_1")

    assert other_epoch.inputs == ["a", "b"]


def test_a_numbered_corpus_grades_and_recalls_like_a_named_one(run):

    first = run(FakeModel(answers={"2": {"passed": False}}), 1, 2, 3)

    recalled = FakeModel(answers={"2": {"passed": False}})
    second = run(recalled, 1, 2, 3)

    assert first["Accuracy-Metrics"]["accuracy"] == pytest.approx(2 / 3)
    assert second == first
    assert recalled.inputs == []


def test_a_numbered_entry_that_failed_is_reconciled_against_the_corpus(run):

    results = run(FakeModel(fails=["2"]), 1, 2, 3)

    assert results["Parsing-Metrics"]["failed"] == 1
    assert results["Parsing-Metrics"]["failed_ids"] == ["2"]
