import datetime

import pytest

from chain_checker.baseclasses.corpus.output import ModelOutput
from chain_checker.baseclasses.loop.prediction_file import (
    PREDICTION_MARKER,
    TOKEN_USAGE_MARKER,
    parse_prediction,
    render_prediction,
)


def _round_trip(predicted: ModelOutput) -> ModelOutput:
    return parse_prediction(render_prediction(predicted), "an/entry.txt")


def test_a_saved_prediction_reads_back_as_it_was_saved():
    predicted = ModelOutput(
        "the conversation", {"passed": True, "score": 0.5}, {"total_tokens": 12}
    )

    recalled = _round_trip(predicted)

    assert recalled.get_convo() == "the conversation"
    assert recalled.get_output() == {"passed": True, "score": 0.5}
    assert recalled.get_token_usage() == {"total_tokens": 12}


def test_a_multi_line_conversation_keeps_every_line():

    conversation = "=== call 1/1 ===\n--- system ---\nbe brief\n\n"

    assert _round_trip(ModelOutput(conversation, {})).get_convo() == conversation


def test_a_call_that_reported_no_usage_saves_no_usage_section():

    rendered = render_prediction(ModelOutput("the conversation", {"passed": True}))

    assert TOKEN_USAGE_MARKER not in rendered
    assert parse_prediction(rendered, "an/entry.txt").get_token_usage() == {}


def test_the_prediction_is_the_models_raw_output_not_the_graded_shape():

    predicted = ModelOutput("", {"passed": True, "notes": ["one", "two"]})

    assert _round_trip(predicted).get_output()["notes"] == ["one", "two"]


def test_an_empty_prediction_is_still_a_readable_file():
    assert _round_trip(ModelOutput("", {})).get_output() == {}


def test_a_value_json_cannot_hold_is_saved_as_text_rather_than_failing():

    predicted = ModelOutput("", {"when": datetime.date(2026, 1, 2)})

    assert _round_trip(predicted).get_output() == {"when": "2026-01-02"}


def test_an_output_mixing_int_and_str_keys_is_still_saved():

    predicted = ModelOutput("", {1: "first", "note": "kept"})

    assert _round_trip(predicted).get_output() == {"1": "first", "note": "kept"}


def test_a_file_with_no_prediction_marker_names_the_file():
    with pytest.raises(ValueError, match="an/entry.txt"):
        parse_prediction("just a conversation", "an/entry.txt")


def test_a_half_written_prediction_names_the_file_and_the_section():

    content = f"the conversation{PREDICTION_MARKER}" + '{"passed": tr'

    with pytest.raises(ValueError, match="prediction JSON in 'an/entry.txt'"):
        parse_prediction(content, "an/entry.txt")


def test_a_broken_usage_section_names_that_section_instead():
    content = (
        f"the conversation{PREDICTION_MARKER}"
        + '{"passed": true}'
        + f"{TOKEN_USAGE_MARKER}not json at all"
    )

    with pytest.raises(ValueError, match="token usage JSON"):
        parse_prediction(content, "an/entry.txt")


@pytest.mark.parametrize(
    "raw",
    ["[1, 2]", '"just text"', "3", "null"],
    ids=["list", "str", "int", "null"],
)
def test_a_prediction_that_is_not_a_json_object_is_refused(raw):

    content = f"the conversation{PREDICTION_MARKER}{raw}"

    with pytest.raises(ValueError, match="not a json object"):
        parse_prediction(content, "an/entry.txt")


def test_a_usage_section_that_is_not_a_json_object_is_refused():
    content = (
        f"the conversation{PREDICTION_MARKER}" + '{"passed": true}' + f"{TOKEN_USAGE_MARKER}[1, 2]"
    )

    with pytest.raises(ValueError, match="not a json object"):
        parse_prediction(content, "an/entry.txt")
