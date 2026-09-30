import uuid

import pytest

from chain_checker.baseclasses.chain.analyser import (
    ChainAnalyser,
    _extract_usage,
)


class _Message:
    def __init__(self, usage_metadata=None) -> None:
        self.usage_metadata = usage_metadata


class _Generation:
    def __init__(self, text="a reply", usage_metadata=None, with_message=True) -> None:
        self.text = text
        if with_message:
            self.message = _Message(usage_metadata)


class _Response:
    def __init__(self, generations=None, llm_output=None) -> None:
        self.generations = [[_Generation()]] if generations is None else generations
        self.llm_output = llm_output


def _run_id():
    return uuid.uuid4()


def _record(cap, *, prompt=None, response=None):

    run_id = _run_id()
    if prompt is not None:
        cap.on_chat_model_start({}, [prompt], run_id=run_id)
    cap.on_llm_end(response or _Response(), run_id=run_id)
    return run_id


def test_a_fresh_analyser_has_recorded_nothing():
    assert ChainAnalyser().calls == []


def test_one_call_records_its_prompt_reply_and_model():
    cap = ChainAnalyser()

    _record(
        cap,
        prompt=["the system message"],
        response=_Response(
            generations=[[_Generation(text="the reply")]],
            llm_output={"model_name": "gpt-4o-mini"},
        ),
    )

    call = cap.calls[0]
    assert call.prompt == ["the system message"]
    assert call.reply == "the reply"
    assert call.model_name == "gpt-4o-mini"


def test_calls_accumulate_in_the_order_they_finished():
    cap = ChainAnalyser()

    for text in ("first", "second", "third"):
        _record(cap, prompt=[text], response=_Response([[_Generation(text=text)]]))

    assert [call.reply for call in cap.calls] == ["first", "second", "third"]


def test_two_calls_in_flight_keep_their_own_prompts():

    cap = ChainAnalyser()
    first, second = _run_id(), _run_id()

    cap.on_chat_model_start({}, [["prompt one"]], run_id=first)
    cap.on_chat_model_start({}, [["prompt two"]], run_id=second)
    cap.on_llm_end(_Response(), run_id=second)
    cap.on_llm_end(_Response(), run_id=first)

    assert [call.prompt for call in cap.calls] == [["prompt two"], ["prompt one"]]


def test_a_finished_call_stops_holding_its_prompt():

    cap = ChainAnalyser()

    _record(cap, prompt=["the system message"])

    assert cap._prompts == {}


def test_a_call_that_failed_stops_holding_its_prompt_too():

    cap = ChainAnalyser()
    run_id = _run_id()

    cap.on_chat_model_start({}, [["the system message"]], run_id=run_id)
    cap.on_llm_error(RuntimeError("rate limited"), run_id=run_id)

    assert cap._prompts == {}


def test_a_call_that_failed_is_not_recorded_as_a_call():

    cap = ChainAnalyser()

    cap.on_llm_error(RuntimeError("rate limited"), run_id=_run_id())

    assert cap.calls == []


def test_an_end_without_a_matching_start_records_an_empty_prompt():
    cap = ChainAnalyser()

    cap.on_llm_end(_Response(), run_id=_run_id())

    assert cap.calls[0].prompt == []


@pytest.mark.parametrize(
    "llm_output",
    [None, {}, {"token_usage": {}}, {"model_name": None}],
    ids=["no-llm-output", "empty-llm-output", "no-model-name-key", "null-model-name"],
)
def test_a_response_that_names_no_model_records_an_empty_name(llm_output):

    cap = ChainAnalyser()

    _record(cap, response=_Response(llm_output=llm_output))

    assert cap.calls[0].model_name == ""


@pytest.mark.parametrize("generations", [[], [[]]], ids=["no-list", "empty-list"])
def test_a_response_with_no_generation_says_what_went_wrong(generations):

    cap = ChainAnalyser()

    with pytest.raises(ValueError, match="no generation"):
        cap.on_llm_end(_Response(generations=generations), run_id=_run_id())


def test_a_start_that_carries_no_messages_records_an_empty_prompt():
    cap = ChainAnalyser()
    run_id = _run_id()

    cap.on_chat_model_start({}, [], run_id=run_id)
    cap.on_llm_end(_Response(), run_id=run_id)

    assert cap.calls[0].prompt == []


def test_a_completion_models_prompt_is_captured_too():
    cap = ChainAnalyser()
    run_id = _run_id()

    cap.on_llm_start({}, ["the raw prompt"], run_id=run_id)
    cap.on_llm_end(_Response(), run_id=run_id)

    assert cap.calls[0].prompt[0].content == "the raw prompt"


def test_a_captured_prompt_reads_the_same_whichever_kind_of_model_sent_it():

    cap = ChainAnalyser()
    run_id = _run_id()

    cap.on_llm_start({}, ["the raw prompt"], run_id=run_id)

    message = cap._prompts[run_id][0]
    assert (message.type, message.content) == ("prompt", "the raw prompt")


def test_usage_metadata_on_the_message_is_preferred():
    generation = _Generation(
        usage_metadata={"input_tokens": 11, "output_tokens": 5, "total_tokens": 16}
    )
    response = _Response(llm_output={"token_usage": {"prompt_tokens": 999}})

    assert _extract_usage(response, generation) == {
        "prompt_tokens": 11,
        "completion_tokens": 5,
        "total_tokens": 16,
    }


def test_llm_output_token_usage_is_read_when_the_message_carries_none():
    response = _Response(
        llm_output={
            "token_usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            }
        }
    )

    assert _extract_usage(response, _Generation()) == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }


def test_a_count_reported_as_null_falls_through_to_the_other_shape():

    generation = _Generation(usage_metadata={"input_tokens": None, "output_tokens": None})
    response = _Response(llm_output={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3}})

    assert _extract_usage(response, generation) == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }


def test_a_missing_total_is_the_sum_of_the_two_halves():
    generation = _Generation(usage_metadata={"input_tokens": 8, "output_tokens": 4})

    assert _extract_usage(_Response(), generation)["total_tokens"] == 12


def test_a_provider_reporting_nothing_costs_zero_rather_than_none():

    assert _extract_usage(_Response(), _Generation(with_message=False)) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def test_counts_reported_as_null_in_both_shapes_are_read_as_zero():
    generation = _Generation(usage_metadata={"input_tokens": None, "output_tokens": None})
    response = _Response(
        llm_output={"token_usage": {"prompt_tokens": None, "completion_tokens": None}}
    )

    assert _extract_usage(response, generation) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def test_a_real_zero_count_is_kept_as_zero():

    generation = _Generation(usage_metadata={"input_tokens": 0, "output_tokens": 4})
    response = _Response(llm_output={"token_usage": {"prompt_tokens": 999}})

    assert _extract_usage(response, generation)["prompt_tokens"] == 0


def test_usage_reaches_the_recorded_call():
    cap = ChainAnalyser()
    generation = _Generation(
        usage_metadata={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
    )

    _record(cap, response=_Response(generations=[[generation]]))

    assert cap.calls[0].token_usage["total_tokens"] == 3
