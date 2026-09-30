import re

from chain_checker.utils.console import link_print_warning

_MARKER_RE = re.compile(
    r"<-*\s*(NEW-LLM-PROMPT|LLM-PROMPT|ACCURACY-REPORT|FALSE-EXAMPLES-REPORT|PARSE-FAILURE-REPORT)\s*-*>",
    re.IGNORECASE,
)

# Also matches a closing `</...>` form, unlike _MARKER_RE: used to strip a
# marker the model echoed back by mistake, not to locate a real one.
_STRAY_MARKER_RE = re.compile(
    r"<\/?-*\s*(NEW-LLM-PROMPT|LLM-PROMPT|ACCURACY-REPORT|FALSE-EXAMPLES-REPORT|PARSE-FAILURE-REPORT)\s*-*>",
    re.IGNORECASE,
)

# Strips a leading think block even without its opening tag: the reply can
# arrive already mid-thought if streaming dropped the opening "<think>".
# clean_candidate() applies this with count=1, so a paired block later in
# the text is left for _THINK_PAIR_RE below.
_THINK_BLOCK_RE = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)

_THINK_PAIR_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Some models wrap plain-text output in a markdown code fence though it
# isn't code. Anchored to the whole text (\A...\Z), not per-line: a fence
# genuinely used inside the new prompt itself (e.g. to show the target
# model's required JSON shape, per MANDATORY 1) must survive untouched.
_CODE_FENCE_RE = re.compile(r"\A```[a-zA-Z]*\n?(.*)\n?```\s*\Z", re.DOTALL)


def clean_candidate(text: str) -> str:
    text = _STRAY_MARKER_RE.sub("", text)
    text = _THINK_BLOCK_RE.sub("", text, count=1)
    text = _THINK_PAIR_RE.sub("", text)
    text = _CODE_FENCE_RE.sub(r"\1", text)
    return text.strip()


def _find_new_prompt_marker(parts: list[str]) -> int | None:
    # _MARKER_RE.split() interleaves marker names at odd indices; walk them
    # back to front so the LAST NEW-LLM-PROMPT marker wins, in case the model
    # echoed an earlier one before writing its real answer after a second.
    for i in range(len(parts) - 2, -1, -2):
        if parts[i].upper() == "NEW-LLM-PROMPT":
            return i
    return None


def _text_after_marker(parts: list[str], marker_index: int) -> str:
    text = clean_candidate(parts[marker_index + 1])
    if not text and marker_index > 0:
        # The model sometimes echoes the marker back before its answer
        # instead of using it as a lead-in - the answer then sits just
        # before the marker rather than after it.
        text = clean_candidate(parts[marker_index - 1])
    return text


def extract_new_prompt(raw_output: str, fallback_prompt: str) -> str:
    """Pulls the rewritten prompt out of the modifier's raw reply: the text
    after its NEW-LLM-PROMPT marker, or the whole cleaned reply if the
    marker is missing. Falls back to `fallback_prompt` - this epoch's prompt,
    unchanged - if nothing usable survives cleanup, rather than ever apply a
    corrupted or empty prompt to the next epoch."""
    parts = _MARKER_RE.split(raw_output)
    marker_index = _find_new_prompt_marker(parts)

    text = (
        _text_after_marker(parts, marker_index)
        if marker_index is not None
        else clean_candidate(raw_output)
    )

    if text:
        return text

    link_print_warning(
        "(R)-(MODIFIER) WARNING: no usable new prompt found in the "
        "modifier's reply (see modifier_output.txt in this epoch's "
        "save_dir) - keeping this epoch's prompt unchanged rather than "
        "risk applying a corrupted one. If this keeps happening, the "
        "modifier model isn't reliably following the marker/format "
        "instructions."
    )
    return fallback_prompt
