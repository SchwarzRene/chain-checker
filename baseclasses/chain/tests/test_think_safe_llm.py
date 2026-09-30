import asyncio
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from chain_checker.baseclasses.chain.analyser import ChainAnalyser
from chain_checker.baseclasses.chain.think_safe_llm import (
    StructuredJSONParseError,
    make_think_safe,
    parse_candidates,
    strip_reasoning,
)


class _Dummy(BaseModel):
    passed: bool


class _Evidence(BaseModel):
    evidence: str
    passed: bool


def _fake_llm(reply: str) -> FakeListChatModel:
    return make_think_safe(FakeListChatModel(responses=[reply]))


# --------------------------------------------------------------------------
# strip_reasoning: the actual text transform, independent of any LLM.
# --------------------------------------------------------------------------


def test_strips_a_well_formed_think_pair():
    assert strip_reasoning("<think>reasoning</think>{}") == "{}"


def test_strips_a_leading_block_missing_its_opening_tag():
    # Observed directly from a real qwen3.5:0.8b reply: the model's own
    # rough-draft JSON came first, then a bare </think>, then the real one.
    reply = '{"draft": true}\n</think>\n\n{"passed": true}'
    assert strip_reasoning(reply) == '{"passed": true}'


def test_strips_a_wrapping_code_fence_left_after_the_think_block():
    assert strip_reasoning('<think>x</think>\n```json\n{"passed": true}\n```') == '{"passed": true}'


def test_a_reply_with_no_think_block_at_all_passes_through_unchanged():
    assert strip_reasoning('{"passed": true}') == '{"passed": true}'


# --------------------------------------------------------------------------
# parse_candidates: which readings of a reply get tried, and in what order.
# --------------------------------------------------------------------------


def test_a_clean_reply_is_the_only_reading_offered():
    assert parse_candidates('{"passed": true}') == ['{"passed": true}']


def test_the_untouched_reply_is_offered_before_the_stripped_one():
    assert parse_candidates('<think>x</think>{"passed": true}') == [
        '<think>x</think>{"passed": true}',
        '{"passed": true}',
    ]


def test_a_broken_quoting_reply_offers_a_repaired_reading_last():
    assert parse_candidates('{"evidence": "the "brand" is uppercase"}') == [
        '{"evidence": "the "brand" is uppercase"}',
        r'{"evidence": "the \"brand\" is uppercase"}',
    ]


def test_an_empty_reply_still_offers_one_reading():
    assert parse_candidates("") == [""]


# --------------------------------------------------------------------------
# make_think_safe + with_structured_output(method="json_mode"): the actual
# safety net, end to end.
# --------------------------------------------------------------------------


def test_json_mode_parses_the_reply_after_stripping_a_think_block():
    llm = _fake_llm('<think>reasoning</think>{"passed": true}')

    result = asyncio.run(llm.with_structured_output(_Dummy, method="json_mode").ainvoke("hi"))

    assert result == _Dummy(passed=True)


def test_json_mode_parses_a_reply_that_never_thought_at_all():
    llm = _fake_llm('{"passed": true}')

    result = asyncio.run(llm.with_structured_output(_Dummy, method="json_mode").ainvoke("hi"))

    assert result == _Dummy(passed=True)


def test_json_mode_recovers_a_reply_whose_only_fault_is_unescaped_quotes():
    # A real ollama failure: the answer is exactly right, but the quotes
    # around the quoted evidence were never escaped, so the reply used to be
    # dropped as an empty prediction.
    llm = _fake_llm('{"evidence": "correct: "100% ACME", "ACME Partner".", "passed": true}')

    result = asyncio.run(llm.with_structured_output(_Evidence, method="json_mode").ainvoke("hi"))

    assert result == _Evidence(evidence='correct: "100% ACME", "ACME Partner".', passed=True)


def test_json_mode_on_unparseable_text_raises_with_the_cleaned_reply_attached():
    llm = _fake_llm("not json at all")

    with pytest.raises(StructuredJSONParseError) as excinfo:
        asyncio.run(llm.with_structured_output(_Dummy, method="json_mode").ainvoke("hi"))

    assert excinfo.value.cleaned_reply == "not json at all"


def test_json_mode_reports_the_last_reading_it_tried_when_all_of_them_fail():
    # Repairing the quoting makes the JSON legal but the schema still isn't
    # satisfied, so the saved text must be what was actually parsed last.
    llm = _fake_llm('<think>x</think>{"wrong": "a "quoted" word"}')

    with pytest.raises(StructuredJSONParseError) as excinfo:
        asyncio.run(llm.with_structured_output(_Dummy, method="json_mode").ainvoke("hi"))

    assert excinfo.value.cleaned_reply == r'{"wrong": "a \"quoted\" word"}'
    assert "validation error" in str(excinfo.value)


def test_json_mode_keeps_pydantics_full_breakdown_on_the_cause():
    llm = _fake_llm('{"passed": "not a bool"}')

    with pytest.raises(StructuredJSONParseError) as excinfo:
        asyncio.run(llm.with_structured_output(_Dummy, method="json_mode").ainvoke("hi"))

    # The message itself stays to one line; the detail lives on __cause__.
    assert len(str(excinfo.value).splitlines()) == 1
    assert excinfo.value.__cause__ is not None


def test_json_mode_parses_a_reply_whose_content_is_a_list_of_blocks(monkeypatch):
    # FakeListChatModel (used by every other test here) can only ever
    # produce string content, so it can't exercise this: langchain-core 1.x
    # lets AIMessage.content be a list of content blocks instead of a plain
    # str (e.g. Responses-API-shaped output), which `.text` - not `.content`
    # - normalizes back to a string.
    llm = make_think_safe(FakeListChatModel(responses=["unused"]))
    reply = AIMessage(content=[{"type": "text", "text": '{"passed": true}'}])
    # Patched on the class, not the instance: llm is a pydantic model, which
    # rejects assigning an attribute that isn't a declared field (see
    # make_think_safe's own docstring on this).
    monkeypatch.setattr(type(llm), "ainvoke", AsyncMock(return_value=reply))

    result = asyncio.run(llm.with_structured_output(_Dummy, method="json_mode").ainvoke("hi"))

    assert result == _Dummy(passed=True)


def test_a_non_json_mode_method_is_left_completely_untouched():
    # FakeListChatModel has no real with_structured_output() of its own, so
    # this must fail exactly as if make_think_safe had never touched llm -
    # confirming method="function_calling" (and friends) fall straight
    # through to the wrapped model's own implementation via super().
    llm = _fake_llm("irrelevant")

    with pytest.raises(NotImplementedError):
        llm.with_structured_output(_Dummy, method="function_calling")


def test_the_raw_reply_is_still_captured_by_a_chain_analyser_callback():
    # The whole point of doing this at the LLM layer rather than inside a
    # chain's own build(): ChainAnalyser (which every chain_checker run
    # wires in) must keep seeing the real raw reply, think block included,
    # exactly as it did before - json_mode's own grammar-constrained
    # decoding is what's being bypassed, not the callback machinery.
    llm = _fake_llm('<think>reasoning</think>{"passed": true}')
    cap = ChainAnalyser()

    asyncio.run(
        llm.with_structured_output(_Dummy, method="json_mode").ainvoke(
            "hi", config={"callbacks": [cap]}
        )
    )

    assert len(cap.calls) == 1
    assert cap.calls[0].reply == '<think>reasoning</think>{"passed": true}'


def test_wrapping_a_plain_object_that_cannot_swap_class_is_a_no_op():
    # Not every "llm" handed in supports a __class__ swap (immutable
    # builtins, some C-extension types) - those also have no real
    # with_structured_output() to make safe, so leaving them untouched
    # (rather than raising) is correct, e.g. for a fake used in another
    # test's app-config double.
    assert make_think_safe("just a string") == "just a string"


def test_the_same_underlying_class_is_only_ever_synthesized_once():
    from chain_checker.baseclasses.chain import think_safe_llm

    llm_a = make_think_safe(FakeListChatModel(responses=["a"]))
    llm_b = make_think_safe(FakeListChatModel(responses=["b"]))

    assert type(llm_a) is type(llm_b)
    assert type(llm_a) in think_safe_llm._safe_classes.values()
