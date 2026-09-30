import pytest

from chain_checker.baseclasses.chain.calls import (
    USAGE_KEYS,
    Call,
    answering_models,
    render_conversation,
    sum_token_usage,
)


class _Message:
    def __init__(self, type, content) -> None:
        self.type = type
        self.content = content


def _call(prompt=(), reply="a reply", model_name="gpt-4o-mini", usage=None) -> Call:
    return Call(list(prompt), reply, model_name, usage or {})


def test_a_call_names_the_four_things_a_reader_needs():

    call = Call(["the prompt"], "the reply", "gpt-4o-mini", {"total_tokens": 3})

    assert call.prompt == ["the prompt"]
    assert call.reply == "the reply"
    assert call.model_name == "gpt-4o-mini"
    assert call.token_usage == {"total_tokens": 3}


def test_a_conversation_names_the_model_every_message_and_the_reply():
    calls = [_call(prompt=[_Message("system", "be brief")], reply="ok")]

    conversation = render_conversation(calls)

    assert "=== call 1/1 ===" in conversation
    assert "--- model ---\ngpt-4o-mini" in conversation
    assert "--- system ---\nbe brief" in conversation
    assert "--- raw reply ---\nok" in conversation


def test_every_call_is_numbered_against_the_total():
    conversation = render_conversation([_call(reply="first"), _call(reply="second")])

    assert "=== call 1/2 ===" in conversation
    assert "=== call 2/2 ===" in conversation


def test_a_multi_turn_prompt_keeps_its_messages_in_order():
    calls = [_call(prompt=[_Message("system", "rules"), _Message("human", "the text")])]

    conversation = render_conversation(calls)

    assert conversation.index("rules") < conversation.index("the text")


def test_the_calls_appear_in_the_order_they_were_captured():
    conversation = render_conversation([_call(reply="first"), _call(reply="second")])

    assert conversation.index("first") < conversation.index("second")


def test_a_chain_that_called_no_model_produces_an_empty_conversation():

    assert render_conversation([]) == ""


def test_usage_is_summed_across_every_call_an_entry_made():
    calls = [
        _call(usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}),
        _call(usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}),
    ]

    assert sum_token_usage(calls) == {
        "prompt_tokens": 15,
        "completion_tokens": 5,
        "total_tokens": 20,
    }


def test_a_call_reporting_no_usage_counts_as_zero_rather_than_breaking_the_sum():
    calls = [_call(usage={"total_tokens": 9}), _call(usage={})]

    assert sum_token_usage(calls) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 9,
    }


def test_an_entry_that_called_no_model_reports_zero_not_an_absent_figure():

    assert sum_token_usage([]) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


@pytest.mark.parametrize("key", USAGE_KEYS)
def test_all_three_counts_are_always_reported(key):

    assert key in sum_token_usage([_call(usage={"total_tokens": 1})])


def test_a_key_no_consumer_reads_is_left_out():

    usage = sum_token_usage([_call(usage={"reasoning_tokens": 7, "total_tokens": 1})])

    assert "reasoning_tokens" not in usage


def test_the_summed_usage_is_not_the_single_calls_own_dict():

    call = _call(usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})

    summed = sum_token_usage([call])
    summed["total_tokens"] = 99999

    assert call.token_usage["total_tokens"] == 2


def test_one_model_answering_every_call_is_named_once():
    assert answering_models([_call(model_name="gpt-4o-mini")] * 3) == ["gpt-4o-mini"]


def test_a_chain_routing_across_models_names_all_of_them_in_a_stable_order():

    calls = [_call(model_name="gpt-4o"), _call(model_name="claude-haiku")]

    assert answering_models(calls) == ["claude-haiku", "gpt-4o"]


def test_a_call_that_named_no_model_adds_nothing():
    calls = [_call(model_name=""), _call(model_name="gpt-4o")]

    assert answering_models(calls) == ["gpt-4o"]


def test_an_entry_that_called_no_model_names_none():
    assert answering_models([]) == []
