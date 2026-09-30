import pytest

from chain_checker.modifier.llm.llm_baseclass import LLM
from chain_checker.utils.errors import CheckerError


def test_a_fresh_instance_reports_no_usage_yet():
    assert LLM().get_last_usage() == {}


def test_the_base_class_refuses_to_generate_anything():
    with pytest.raises(CheckerError, match="abstract base class"):
        LLM()("some prompt")


def test_the_base_class_reports_no_run_info():
    assert LLM().get_run_info() == {}


def test_get_last_usage_hands_back_a_copy_not_the_live_dict():
    llm = LLM()
    llm._record_usage(1, 2, 3)

    leaked = llm.get_last_usage()
    leaked["prompt_tokens"] = 999

    assert llm.get_last_usage() == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}


@pytest.mark.parametrize(
    "prompt_tokens, completion_tokens, total_tokens, expected_total",
    [
        (10, 5, None, 15),
        (10, 5, 20, 20),
        (0, 0, None, 0),
    ],
    ids=["derives-total-when-missing", "keeps-a-reported-total", "empty-is-a-real-zero"],
)
def test_record_usage_derives_a_missing_total_but_trusts_a_given_one(
    prompt_tokens, completion_tokens, total_tokens, expected_total
):
    llm = LLM()
    llm._record_usage(prompt_tokens, completion_tokens, total_tokens)

    assert llm.get_last_usage() == {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": expected_total,
    }


def test_require_nonempty_text_passes_through_real_text():
    assert LLM()._require_nonempty_text("hello", "should not fire") == "hello"


def test_require_nonempty_text_fails_on_empty_text():
    with pytest.raises(CheckerError, match="no reply at all"):
        LLM()._require_nonempty_text("", "no reply at all")
