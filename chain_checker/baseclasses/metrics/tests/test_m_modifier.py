"""Unit tests for m_modifier.py.

ModifierMetrics is the assembly point: it walks the corpus once, feeds every
metric, and returns the list the prompt modifier is handed. The list's order
and the metrics' names are the interface - the modifier picks the
false-example metric out of it by name.
"""

from chain_checker.baseclasses.metrics.m_modifier import ModifierMetrics


def _by_name(metrics: list) -> dict:
    return {metric["name"]: metric for metric in metrics}


def test_every_metric_is_reported_in_the_base_to_dict_shape(make_entry, make_corpus):
    corpus = make_corpus(make_entry("a"))

    for metric in ModifierMetrics(corpus).get_metrics():
        assert set(metric) == {"name", "description", "results"}


def test_the_false_example_metric_is_findable_by_the_name_modify_looks_for(make_entry, make_corpus):
    # The modifier picks this one out by name and treats the rest as the
    # accuracy report, so renaming it silently empties its evidence.
    corpus = make_corpus(make_entry("a", expected={"passed": True}, predicted={"passed": False}))

    names = _by_name(ModifierMetrics(corpus).get_metrics())

    assert "Negative-Predicted-Metrics" in names
    assert names["Negative-Predicted-Metrics"]["results"]["a"]["matches"] == {"passed": False}


def test_the_full_set_of_metrics_is_returned_every_time(make_entry, make_corpus):
    corpus = make_corpus(make_entry("a"))

    names = list(_by_name(ModifierMetrics(corpus).get_metrics()))

    assert names == [
        "Accuracy-Metrics",
        "Negative-Predicted-Metrics",
        "Chain-Token-Usage-Metrics",
        "Parsing-Metrics",
        "Modification-Metrics",
        "TextLength-Metrics",
        "Language-Metrics",
        "Labeller-Metrics",
    ]


def test_an_empty_corpus_still_reports_every_metric(make_corpus):
    # The training loop renders a report unconditionally; nothing here may
    # assume at least one entry.
    metrics = _by_name(ModifierMetrics(make_corpus()).get_metrics())

    assert metrics["Accuracy-Metrics"]["results"] == {"rules": {}, "accuracy": 0.0}
    assert metrics["Parsing-Metrics"]["results"]["total"] == 0
    assert metrics["Language-Metrics"]["results"] == {}


def test_info_breakdowns_are_built_from_the_entries_that_carry_them(make_entry, make_corpus):
    corpus = make_corpus(
        make_entry("a", info={"language": "en", "labeller": "alice", "modification": "swap"}),
        make_entry("b", info={"language": "en", "labeller": "bob", "modification": "typo"}),
        make_entry("c", info={"language": "de", "labeller": "alice", "modification": "swap"}),
    )

    results = _by_name(ModifierMetrics(corpus).get_metrics())

    assert results["Language-Metrics"]["results"] == {"en": 2, "de": 1}
    assert results["Labeller-Metrics"]["results"] == {"alice": 2, "bob": 1}
    assert results["Modification-Metrics"]["results"] == ["swap", "typo", "swap"]


def test_a_corpus_with_no_info_gets_empty_breakdowns_rather_than_a_crash(make_entry, make_corpus):
    # info is free-form and optional, so a corpus carrying none is normal.
    corpus = make_corpus(make_entry("a"), make_entry("b"))

    results = _by_name(ModifierMetrics(corpus).get_metrics())

    assert results["Language-Metrics"]["results"] == {}
    assert results["Modification-Metrics"]["results"] == []


def test_an_entry_missing_only_some_info_contributes_only_what_it_has(make_entry, make_corpus):
    # `if x is not None` per field, not one all-or-nothing guard.
    corpus = make_corpus(
        make_entry("a", info={"language": "en"}),
        make_entry("b", info={"labeller": "bob"}),
    )

    results = _by_name(ModifierMetrics(corpus).get_metrics())

    assert results["Language-Metrics"]["results"] == {"en": 1}
    assert results["Labeller-Metrics"]["results"] == {"bob": 1}


def test_an_unhashable_info_value_is_skipped_rather_than_killing_the_epoch(make_entry, make_corpus):
    # KeywordMetrics uses the value as a dict key, so a list raises TypeError
    # inside get_metrics() - after the corpus has been paid for. info is
    # free-form, so `language: [en, de]` is writable.
    corpus = make_corpus(
        make_entry("a", info={"language": ["en", "de"], "labeller": {"name": "alice"}}),
        make_entry("b", info={"language": "en", "labeller": "bob"}),
    )

    results = _by_name(ModifierMetrics(corpus).get_metrics())

    assert results["Language-Metrics"]["results"] == {"en": 1}
    assert results["Labeller-Metrics"]["results"] == {"bob": 1}


def test_an_unhashable_modification_still_reaches_the_list_metric(make_entry, make_corpus):
    # ListMetrics appends rather than counting, so the hashability guard is on
    # the two KeywordMetrics feeds only and must not narrow this one.
    corpus = make_corpus(make_entry("a", info={"modification": ["swap", "typo"]}))

    results = _by_name(ModifierMetrics(corpus).get_metrics())

    assert results["Modification-Metrics"]["results"] == [["swap", "typo"]]


def test_text_lengths_are_word_counts_of_the_input_text(make_entry, make_corpus):
    corpus = make_corpus(
        make_entry("a", text="one two three"),
        make_entry("b", text="single"),
    )

    results = _by_name(ModifierMetrics(corpus).get_metrics())

    assert results["TextLength-Metrics"]["results"] == [3, 1]


def test_a_chain_whose_input_has_no_text_field_contributes_no_lengths(make_corpus):
    # "text" isn't a guaranteed InputSchema field across every chain.
    from chain_checker.baseclasses.corpus.entry import Entry
    from chain_checker.baseclasses.corpus.input import Input
    from chain_checker.baseclasses.corpus.label import Label
    from chain_checker.baseclasses.corpus.output import ModelOutput

    corpus = make_corpus(
        Entry(
            "a",
            Input({"document": "no text key here"}),
            Label({"passed": True}),
            {},
            ModelOutput("", {}),
        )
    )

    results = _by_name(ModifierMetrics(corpus).get_metrics())

    assert results["TextLength-Metrics"]["results"] == []


def test_the_failed_ids_reach_the_parsing_metric(make_entry, make_corpus):
    corpus = make_corpus(make_entry("a"), make_entry("b"), make_entry("c"))

    results = _by_name(ModifierMetrics(corpus, failed_ids={"b"}).get_metrics())

    assert results["Parsing-Metrics"]["results"] == {
        "total": 3,
        "parsed": 2,
        "failed": 1,
        "failure_rate": 1 / 3,
        "failed_ids": ["b"],
    }


def test_a_failed_id_the_corpus_does_not_hold_is_left_out(make_entry, make_corpus):
    # 'parsed' is total minus failed, so an unreconciled stray id reports a
    # negative count and a failure rate above 1.0 - both of which the report
    # renders verbatim and the modifier reads as fact. The corpus is in hand
    # here, so the reconciliation belongs here rather than in ParsingMetrics.
    corpus = make_corpus(make_entry("a"), make_entry("b"))

    results = _by_name(
        ModifierMetrics(corpus, failed_ids={"a", "not-in-the-corpus"}).get_metrics()
    )["Parsing-Metrics"]["results"]

    assert results == {
        "total": 2,
        "parsed": 1,
        "failed": 1,
        "failure_rate": 0.5,
        "failed_ids": ["a"],
    }


def test_a_parsing_report_never_claims_more_failures_than_entries(make_entry, make_corpus):
    # The invariant the reconciliation buys, stated on its own so a later
    # change to how failed_ids are passed has to keep it.
    corpus = make_corpus(make_entry("a"))

    results = _by_name(ModifierMetrics(corpus, failed_ids={"a", "b", "c"}).get_metrics())[
        "Parsing-Metrics"
    ]["results"]

    assert results["parsed"] >= 0
    assert results["failed"] <= results["total"]
    assert 0.0 <= results["failure_rate"] <= 1.0


def test_no_failed_ids_defaults_to_a_clean_parsing_report(make_entry, make_corpus):
    corpus = make_corpus(make_entry("a"))

    results = _by_name(ModifierMetrics(corpus).get_metrics())

    assert results["Parsing-Metrics"]["results"]["failed"] == 0


def test_the_failed_ids_are_copied_rather_than_held_by_reference(make_entry, make_corpus):
    # The epoch's report must not drift if the caller keeps adding to the
    # collection it passed, whatever that caller's own copying does.
    corpus = make_corpus(make_entry("a"), make_entry("b"))
    failed = {"a"}
    metrics = ModifierMetrics(corpus, failed_ids=failed)

    failed.add("b")

    results = _by_name(metrics.get_metrics())["Parsing-Metrics"]["results"]

    assert results["failed_ids"] == ["a"]
    assert results["failed"] == 1


def test_get_metrics_builds_fresh_metrics_on_every_call(make_entry, make_corpus):
    # Called once per epoch against a re-run corpus: reusing the metric objects
    # would accumulate every epoch into the first one's numbers.
    corpus = make_corpus(make_entry("a", expected={"passed": True}, predicted={"passed": True}))

    first = _by_name(ModifierMetrics(corpus).get_metrics())
    second = _by_name(ModifierMetrics(corpus).get_metrics())

    assert first["Accuracy-Metrics"]["results"] == second["Accuracy-Metrics"]["results"]
    assert second["Accuracy-Metrics"]["results"]["rules"]["passed"]["results"]["TT"] == 1


def test_the_same_corpus_can_be_assembled_by_two_metrics_independently(make_entry, make_corpus):
    # The assembler holds no state between calls, so a caller can build one per
    # epoch off the same corpus object without the two influencing each other.
    corpus = make_corpus(make_entry("a", expected={"passed": True}, predicted={"passed": False}))

    first = ModifierMetrics(corpus, failed_ids={"a"})
    second = ModifierMetrics(corpus)

    assert _by_name(first.get_metrics())["Parsing-Metrics"]["results"]["failed"] == 1
    assert _by_name(second.get_metrics())["Parsing-Metrics"]["results"]["failed"] == 0
