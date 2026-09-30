import pytest

from chain_checker.modifier.model.extraction import clean_candidate, extract_new_prompt

# --------------------------------------------------------------------------
# clean_candidate: stripping markers, think-blocks and code fences
# --------------------------------------------------------------------------


def test_a_stray_marker_is_removed():
    assert clean_candidate("<---LLM-PROMPT---> the real text") == "the real text"


def test_a_closing_stray_marker_is_removed_too():
    assert clean_candidate("the real text</---LLM-PROMPT--->") == "the real text"


def test_a_leading_unclosed_think_block_is_stripped():
    assert clean_candidate("reasoning about it</think>the answer") == "the answer"


def test_only_the_first_unclosed_think_marker_is_stripped():
    # count=1 in clean_candidate(): a second, real </think> later in the
    # text is left for _THINK_PAIR_RE, not eaten by the leading strip.
    assert clean_candidate("junk</think><think>more junk</think>the answer") == "the answer"


def test_a_properly_paired_think_block_is_stripped():
    assert clean_candidate("<think>reasoning</think>the answer") == "the answer"


def test_a_markdown_code_fence_is_stripped():
    assert clean_candidate("```\nthe answer\n```") == "the answer"


def test_a_language_tagged_code_fence_is_stripped():
    assert clean_candidate("```text\nthe answer\n```") == "the answer"


def test_a_code_fence_that_does_not_wrap_the_whole_reply_survives():
    # A fence used inside the new prompt itself (e.g. to show the target
    # model's required JSON shape) is real content, not wrapper noise -
    # only a fence spanning the entire reply is a wrapper to strip.
    text = 'Respond as:\n```json\n{"a": 1}\n```\nNever add extra text.'
    assert clean_candidate(text) == text


def test_surrounding_whitespace_is_trimmed():
    assert clean_candidate("   the answer   \n") == "the answer"


def test_plain_text_with_nothing_to_clean_is_left_alone():
    assert clean_candidate("just a normal prompt") == "just a normal prompt"


# --------------------------------------------------------------------------
# extract_new_prompt: pulling the rewrite out of the modifier's raw reply
# --------------------------------------------------------------------------


def test_the_text_after_the_marker_is_the_new_prompt():
    raw = "<---NEW-LLM-PROMPT---> Be nice to the user."
    assert extract_new_prompt(raw, fallback_prompt="old") == "Be nice to the user."


def test_a_reply_with_no_marker_at_all_is_used_whole():
    assert extract_new_prompt("Be nice to the user.", fallback_prompt="old") == (
        "Be nice to the user."
    )


def test_reasoning_after_the_marker_is_stripped_from_the_result():
    raw = "<---NEW-LLM-PROMPT---> <think>hmm, let me consider...</think>Be nice."
    assert extract_new_prompt(raw, fallback_prompt="old") == "Be nice."


def test_only_the_last_marker_is_honoured_if_the_model_echoed_an_earlier_one():
    raw = "<---NEW-LLM-PROMPT---> quoted from history <---NEW-LLM-PROMPT---> Be nice to the user."
    assert extract_new_prompt(raw, fallback_prompt="old") == "Be nice to the user."


def test_an_empty_marker_falls_back_to_the_text_just_before_it():
    raw = "Be nice to the user. <---NEW-LLM-PROMPT--->   "
    assert extract_new_prompt(raw, fallback_prompt="old") == "Be nice to the user."


@pytest.mark.parametrize(
    "raw",
    ["<---NEW-LLM-PROMPT--->   ", "", "```\n```", "<think></think>"],
    ids=["blank-after-marker", "empty-reply", "empty-code-fence", "empty-think-block"],
)
def test_a_reply_with_nothing_usable_falls_back_to_the_current_prompt(raw):
    assert extract_new_prompt(raw, fallback_prompt="old prompt") == "old prompt"


def test_a_reply_with_nothing_usable_warns_on_stdout(capsys):
    extract_new_prompt("", fallback_prompt="old prompt")

    assert "WARNING" in capsys.readouterr().out
