"""Unit tests for chain_checker/baseclasses/metrics/m_keyword.py.

KeywordMetrics scores nothing. It describes the corpus - what languages it
holds, who labelled it - so the modifier LLM can see the shape of the data it
is being asked to write a prompt for. Fed through add_item().
"""

import json

import yaml

from chain_checker.baseclasses.metrics.m_keyword import KeywordMetrics


def test_an_unseen_corpus_reports_an_empty_breakdown():
    # ModifierMetrics builds these unconditionally, so a corpus with no
    # language/labeller info has to render as {} rather than crash.
    assert KeywordMetrics("Language", "d").compute() == {}


def test_each_item_is_counted_once_per_occurrence():
    metric = KeywordMetrics("Language", "d")
    for language in ("en", "de", "en", "en"):
        metric.add_item(language)

    assert metric.compute() == {"en": 3, "de": 1}


def test_counts_are_kept_per_key_not_pooled():
    metric = KeywordMetrics("Labeller", "d")
    metric.add_item("alice")
    metric.add_item("bob")

    assert metric.compute() == {"alice": 1, "bob": 1}


def test_a_falsy_key_is_counted_like_any_other():
    # Counter counts occurrences of a key, so an empty-string or zero label is
    # a category of its own rather than being dropped as falsy.
    metric = KeywordMetrics("Language", "d")
    metric.add_item("")
    metric.add_item(0)
    metric.add_item(False)

    assert metric.compute()[""] == 1
    # 0 and False are the same dict key in Python - one bucket, counted twice.
    assert metric.compute()["0"] == 2


def test_the_reported_keys_are_text_whatever_was_counted():
    # metrics.json is written with sort_keys=True, which orders the keys before
    # coercing them: a set holding two types raises there, at the end of a run
    # that has already been paid for.
    metric = KeywordMetrics("Language", "d")
    metric.add_item("en")
    metric.add_item(False)
    metric.add_item(7)

    assert metric.compute() == {"en": 1, "False": 1, "7": 1}
    assert all(isinstance(key, str) for key in metric.compute())


def test_a_mixed_type_breakdown_still_serialises():
    # The crash this guards, reached the way a corpus reaches it: YAML reads a
    # bare `no` as False, so one Norwegian entry beside an English one is all
    # a free-form `language` field needs.
    metric = KeywordMetrics("Language", "d")
    for language in yaml.safe_load("langs: [en, en, no]")["langs"]:
        metric.add_item(language)

    assert json.dumps(metric.compute(), sort_keys=True) is not None
    assert metric.compute() == {"en": 2, "False": 1}


def test_two_values_sharing_one_text_are_summed_not_overwritten():
    # str() can collapse two Counter buckets into one reported key, and the
    # counts have to survive that rather than the last one winning.
    metric = KeywordMetrics("Labeller", "d")
    metric.add_item(7)
    metric.add_item("7")
    metric.add_item("7")

    assert metric.compute() == {"7": 3}


def test_keyword_counts_survive_being_computed_twice():
    metric = KeywordMetrics("Language", "d")
    metric.add_item("en")

    assert metric.compute() == metric.compute() == {"en": 1}


def test_the_reported_counts_are_a_copy():
    # The results are mutated and serialised downstream, so compute() must not
    # hand out the container it accumulates into.
    metric = KeywordMetrics("Language", "d")
    metric.add_item("en")

    metric.compute()["en"] = 999

    assert metric.compute() == {"en": 1}


def test_the_counts_are_reported_as_a_plain_dict():
    # Counter is an implementation detail; the reported type is what gets
    # serialised and compared downstream.
    metric = KeywordMetrics("Language", "d")
    metric.add_item("en")

    assert type(metric.compute()) is dict
