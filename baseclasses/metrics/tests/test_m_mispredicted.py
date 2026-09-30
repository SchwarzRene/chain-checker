"""Unit tests for m_mispredicted.py.

MispredictedMetrics is graded through Label.matches() instead of its own
comparison, so every case here is really testing that the two agree, and
that the record it keeps is safe to hand to the prompt modifier as evidence.
"""

import json

from chain_checker.baseclasses.metrics.m_mispredicted import MispredictedMetrics


def test_nothing_is_recorded_for_a_run_with_no_mispredictions(make_entry):
    metric = MispredictedMetrics()
    metric.add_entry(make_entry("a", expected={"passed": True}, predicted={"passed": True}))

    assert metric.compute() == {}


def test_an_entry_that_failed_a_checked_key_is_recorded_under_its_id(make_entry):
    metric = MispredictedMetrics()
    metric.add_entry(
        make_entry(
            "case-7",
            expected={"passed": True},
            predicted={"passed": False},
            text="the input text",
        )
    )

    assert metric.compute() == {
        "case-7": {
            "input": {"text": "the input text"},
            "true": {"passed": True},
            "predicted": {"passed": False},
            "matches": {"passed": False},
        }
    }


def test_an_entry_that_failed_only_one_of_several_keys_is_still_recorded(make_entry):
    # One failing key is enough to record the whole case; 'matches' then
    # tells the modifier exactly which key was wrong.
    metric = MispredictedMetrics()
    metric.add_entry(
        make_entry(
            "a",
            expected={"passed": True, "score": 0.9},
            predicted={"passed": True, "score": 0.1},
        )
    )

    assert metric.compute()["a"]["matches"] == {"passed": True, "score": False}


def test_predicted_carries_only_the_keys_the_case_checks(make_entry):
    # A chain's real output can carry far more than the case checks; only the
    # checked keys belong in the evidence, or the real disagreement gets lost
    # in noise.
    metric = MispredictedMetrics()
    metric.add_entry(
        make_entry(
            "a",
            expected={"passed": True},
            predicted={"passed": False, "reasoning": "a long unchecked explanation"},
        )
    )

    assert metric.compute()["a"]["predicted"] == {"passed": False}


def test_a_key_the_chain_never_produced_is_marked_missing_not_none(make_entry):
    # A missing key can't be reported as None: a corpus may legitimately
    # expect null, which would make this same record's 'matches': False look
    # self-contradictory (see the next test).
    metric = MispredictedMetrics()
    metric.add_entry(make_entry("a", expected={"passed": True}, predicted={}))

    assert metric.compute()["a"]["predicted"] == {"passed": "<missing>"}
    assert metric.compute()["a"]["matches"] == {"passed": False}


def test_a_real_predicted_null_is_distinguishable_from_a_missing_key(make_entry):
    # The other half of the previous test: a genuinely predicted null must
    # still read differently from a key that was never produced at all.
    metric = MispredictedMetrics()
    metric.add_entry(
        make_entry(
            "a",
            expected={"reason": None, "passed": True},
            predicted={"reason": None, "passed": False},
        )
    )

    assert metric.compute()["a"]["predicted"] == {"reason": None, "passed": False}


def test_an_unrun_entry_is_recorded_as_a_total_miss(make_entry):
    # What an entry looks like after Corpus.reset() or a parsing failure -
    # nothing predicted, so every checked key fails.
    metric = MispredictedMetrics()
    metric.add_entry(make_entry("a", expected={"passed": True, "score": 0.9}))

    assert metric.compute()["a"]["matches"] == {"passed": False, "score": False}


def test_the_same_id_added_twice_keeps_only_the_later_record(make_entry):
    # Entries are keyed by id, so scoring the same id again overwrites its
    # record instead of appending a second one.
    metric = MispredictedMetrics()
    metric.add_entry(make_entry("a", expected={"passed": True}, predicted={"passed": False}))
    metric.add_entry(make_entry("a", expected={"passed": True}, predicted={"passed": None}))

    assert list(metric.compute()) == ["a"]
    assert metric.compute()["a"]["predicted"] == {"passed": None}


def test_an_entry_can_stop_being_a_counter_example_only_by_being_left_out(make_entry):
    # add_entry() only ever adds a failing record, never removes one - a later
    # passing attempt for the same id does not erase the earlier failure.
    # Clearing this out is the caller's job, one fresh metric per epoch.
    metric = MispredictedMetrics()
    metric.add_entry(make_entry("a", expected={"passed": True}, predicted={"passed": False}))
    metric.add_entry(make_entry("a", expected={"passed": True}, predicted={"passed": True}))

    assert metric.compute()["a"]["predicted"] == {"passed": False}


# --------------------------------------------------------------------------
# Grading through Label.matches() - the semantics come along for free
# --------------------------------------------------------------------------


def test_an_int_prediction_against_a_bool_label_is_a_counter_example(make_entry):
    # matches() keeps bool and int apart the same way values_agree does
    # everywhere else, so this can't disagree with what
    # test_matches_consistency.py pins down.
    metric = MispredictedMetrics()
    metric.add_entry(make_entry("a", expected={"passed": True}, predicted={"passed": 1}))

    assert metric.compute()["a"]["matches"] == {"passed": False}


def test_a_float_correct_to_sixteen_digits_is_not_a_counter_example(make_entry):
    # Floats compare with a tolerance here too, so a binary-rounding artifact
    # like 0.1 + 0.2 doesn't get handed to the modifier as a real mistake.
    metric = MispredictedMetrics()
    metric.add_entry(make_entry("a", expected={"score": 0.3}, predicted={"score": 0.1 + 0.2}))

    assert metric.compute() == {}


def test_a_nested_bool_mistyped_as_an_int_is_a_counter_example(make_entry):
    # The recursion into nested dicts applies the same bool/int strictness at
    # any depth, not just at the top level.
    metric = MispredictedMetrics()
    metric.add_entry(
        make_entry(
            "a",
            expected={"verdicts": {"r1": True}},
            predicted={"verdicts": {"r1": 1}},
        )
    )

    assert metric.compute()["a"]["matches"] == {"verdicts": False}


def test_nested_floats_within_tolerance_are_not_a_counter_example(make_entry):
    metric = MispredictedMetrics()
    metric.add_entry(
        make_entry(
            "a",
            expected={"scores": [0.3]},
            predicted={"scores": [0.1 + 0.2]},
        )
    )

    assert metric.compute() == {}


# --------------------------------------------------------------------------
# Reported state
# --------------------------------------------------------------------------


def test_entries_are_keyed_by_text_even_for_an_int_corpus_id(make_entry):
    # A corpus can mix int and string case ids; metrics.json is written with
    # sort_keys=True, which raises on a dict keyed with both, so ids are
    # normalized to text here.
    metric = MispredictedMetrics()
    metric.add_entry(make_entry(1, expected={"passed": True}, predicted={"passed": False}))
    metric.add_entry(make_entry("b", expected={"passed": True}, predicted={"passed": False}))

    results = metric.compute()

    assert set(results) == {"1", "b"}
    json.dumps(results, sort_keys=True)


def test_the_reported_dict_is_a_copy(make_entry):
    # Only a shallow copy: nothing downstream mutates a per-entry record, so
    # what actually needs protecting is the outer mapping - a caller must not
    # be able to add or remove entries.
    metric = MispredictedMetrics()
    metric.add_entry(make_entry("a", expected={"passed": True}, predicted={"passed": False}))

    metric.compute()["injected"] = {"not": "a real entry"}

    assert "injected" not in metric.compute()


def test_compute_does_not_consume_what_was_added(make_entry):
    metric = MispredictedMetrics()
    metric.add_entry(make_entry("a", expected={"passed": True}, predicted={"passed": False}))

    assert metric.compute() == metric.compute()
