"""Tests for corpus/output.py and corpus/output_empty.py together.

ModelOutput subclasses Label so a prediction can be graded directly against a
corpus label, and Corpus.load()/reset() hand out an EmptyModelOutput - the
reason both classes live in the corpus package rather than in chain. The
payload backing ModelOutput is stored once, inside the Label view's Record;
several tests below exist specifically to pin that set()/set_output() and
get()/get_output() can never see two different values for the same call.
"""

import pytest

from chain_checker.baseclasses.corpus.label import Label
from chain_checker.baseclasses.corpus.output import ModelOutput
from chain_checker.baseclasses.corpus.output_empty import EmptyModelOutput

# --------------------------------------------------------------------------
# Construction and accessors
# --------------------------------------------------------------------------


def test_the_accessors_return_what_was_passed_in():
    output = ModelOutput("the conversation", {"passed": True}, {"total_tokens": 5})

    assert output.get_convo() == "the conversation"
    assert output.get_output() == {"passed": True}
    assert output.get_token_usage() == {"total_tokens": 5}


def test_token_usage_defaults_to_an_empty_dict():
    # token_usage is optional, but the metric that reads it does so
    # unconditionally, so a caller that never measured usage must still get
    # a mapping back rather than None.
    assert ModelOutput("", {}).get_token_usage() == {}
    assert ModelOutput("", {}, None).get_token_usage() == {}


def test_the_constructor_copies_the_token_usage_it_is_handed():
    usage = {"total_tokens": 5}
    output = ModelOutput("", {}, usage)

    usage["total_tokens"] = 999

    assert output.get_token_usage() == {"total_tokens": 5}


def test_get_token_usage_returns_a_copy_not_a_reference():
    # The token-usage metric holds on to what it collects for the whole
    # epoch, so a caller must not be able to rewrite a recorded cost through
    # the getter it was handed.
    output = ModelOutput("", {}, {"total_tokens": 5})

    output.get_token_usage()["total_tokens"] = 999

    assert output.get_token_usage() == {"total_tokens": 5}


def test_the_label_view_is_seeded_from_the_output_at_construction():
    # The constructor hands the output straight to Record, so the inherited
    # Label data reflects the model's output from the start.
    output = ModelOutput("", {"passed": True, "score": 0.9})

    assert output.get() == {"passed": True, "score": 0.9}
    assert output.get_keys() == ["passed", "score"]
    assert output.get_value("score") == 0.9


def test_the_two_views_are_the_same_payload():
    output = ModelOutput("", {"passed": True})

    assert output.get_output() == output.get() == {"passed": True}


def test_str_identifies_the_subclass():
    # Record.__str__ reads self.__class__.__name__, not a hard-coded name.
    assert str(ModelOutput("", {"passed": True})).startswith("ModelOutput:")


# --------------------------------------------------------------------------
# set_convo / set_output
# --------------------------------------------------------------------------


def test_set_convo_replaces_the_conversation():
    output = ModelOutput("first", {})

    output.set_convo("second")

    assert output.get_convo() == "second"


def test_set_output_updates_both_the_raw_output_and_the_label_view():
    # This is what a run relies on: Corpus.reset() hands out empty
    # predictions up front, and the run has to be able to fill them in and
    # have both views agree afterwards.
    output = EmptyModelOutput()
    assert output.get() == {}

    output.set_output({"passed": True})

    assert output.get_output() == {"passed": True}
    assert output.get() == {"passed": True}


# --------------------------------------------------------------------------
# The two views cannot drift apart
# --------------------------------------------------------------------------


def test_the_inherited_label_set_also_updates_what_gets_serialised():
    # set() is inherited from Label and ModelOutput does not shadow it, so
    # calling it directly (instead of set_output()) must still keep the
    # serialised output in step with the graded view.
    output = ModelOutput("convo", {"passed": True})

    output.set({"passed": False})

    assert output.get() == {"passed": False}
    assert output.get_output() == {"passed": False}


def test_set_and_set_output_are_interchangeable():
    via_set, via_set_output = ModelOutput("", {}), ModelOutput("", {})

    via_set.set({"passed": True})
    via_set_output.set_output({"passed": True})

    assert via_set.get() == via_set_output.get()
    assert via_set.get_output() == via_set_output.get_output()


def test_the_output_is_copied_at_construction():
    # Matches Record's own copy-in guarantee: the caller's dict must not go
    # on steering the prediction after construction.
    source = {"passed": True}
    output = ModelOutput("", source)

    source["passed"] = "mutated after construction"

    assert output.get_output() == {"passed": True}
    assert output.get() == {"passed": True}


def test_neither_getter_hands_out_the_live_dict():
    output = ModelOutput("", {"passed": True})

    output.get_output()["passed"] = "discarded with the copy"
    output.get()["ignored"] = "discarded with the copy"

    assert output.get_output() == {"passed": True}
    assert output.get() == {"passed": True}


def test_set_output_copies_the_dict_it_is_given():
    source = {"passed": "original"}
    output = ModelOutput("", {})

    output.set_output(source)
    source["passed"] = "mutated after set_output"

    assert output.get_output() == {"passed": "original"}
    assert output.get() == {"passed": "original"}


# --------------------------------------------------------------------------
# Comparing a prediction against a corpus label
# --------------------------------------------------------------------------


def test_a_prediction_is_a_label_so_a_corpus_label_can_grade_it():
    # This inheritance is the entire point: Label.matches() takes anything
    # exposing get(), and ModelOutput qualifies.
    assert isinstance(ModelOutput("", {}), Label)

    truth = Label({"passed": True, "score": 0.9})
    prediction = ModelOutput("", {"passed": True, "score": 0.9, "extra": "ignored"})

    assert truth.matches(prediction) == {"passed": True, "score": True}


def test_a_corpus_label_reports_the_key_a_prediction_got_wrong():
    truth = Label({"passed": True, "score": 0.9})
    prediction = ModelOutput("", {"passed": True, "score": 0.1})

    assert truth.matches(prediction) == {"passed": True, "score": False}


# --------------------------------------------------------------------------
# EmptyModelOutput
# --------------------------------------------------------------------------


def test_an_empty_prediction_is_blank_on_every_view():
    empty = EmptyModelOutput()

    assert empty.get_convo() == ""
    assert empty.get_output() == {}
    assert empty.get_token_usage() == {}
    assert empty.get() == {}
    assert empty.get_keys() == []


def test_an_empty_prediction_is_a_model_output():
    assert isinstance(EmptyModelOutput(), ModelOutput)
    assert isinstance(EmptyModelOutput(), Label)


def test_every_checked_key_fails_against_an_empty_prediction():
    # This is what an unrun case grades as, which is why Corpus.reset() can
    # safely hand out an EmptyModelOutput instead of leaving the slot unset.
    truth = Label({"passed": True, "score": 0.9})

    assert truth.matches(EmptyModelOutput()) == {"passed": False, "score": False}


def test_two_empty_predictions_do_not_share_state():
    # If the constructor's default argument were a single shared mutable
    # dict, Corpus.reset() would wire every entry to the same prediction.
    first, second = EmptyModelOutput(), EmptyModelOutput()

    first.set_output({"passed": True})

    assert second.get_output() == {}
    assert second.get() == {}


def test_an_empty_prediction_takes_no_constructor_arguments():
    with pytest.raises(TypeError):
        EmptyModelOutput("unexpected")  # type: ignore[call-arg]
