import sys
import types

import pytest

from chain_checker.baseclasses.chain.model import Model


@pytest.fixture
def chain_module():

    module = types.ModuleType("fake_chain_module")
    sys.modules[module.__name__] = module
    try:
        yield module
    finally:
        del sys.modules[module.__name__]


def _model(chain_module, *template_vars) -> Model:
    model = Model.__new__(Model)
    model._chain_type = "template_checklist"
    model._prompt_template_vars = tuple(template_vars)
    model.chain = types.SimpleNamespace(__module__=chain_module.__name__)
    model.rebuilds = []

    def _build_runnable():
        model.rebuilds.append("rebuilt")
        return "a runnable"

    model._build_runnable = _build_runnable
    return model


def test_the_prompt_is_handed_out_as_a_model_sees_it(chain_module):

    chain_module.SYSTEM_PROMPT = 'Rewrite {text}. Reply as {{"ok": true}}'

    assert _model(chain_module, "text").get_system_prompt() == (
        'Rewrite {text}. Reply as {"ok": true}'
    )


def _with_rules_block(chain_module, rules: str = "RULE A\nRULE B") -> None:
    # Mirrors a real leaf chain's generateSystemPrompt: reads its OWN module's
    # current SYSTEM_PROMPT (so a rewrite lands here too) and fills the
    # placeholder from a rule set computed elsewhere.
    def generateSystemPrompt(payload: dict) -> str:
        return chain_module.SYSTEM_PROMPT.format(rules_block=rules)

    chain_module.generateSystemPrompt = generateSystemPrompt


def test_the_prompt_is_handed_out_with_the_chains_placeholders_filled(chain_module):
    # The whole point: a modifier reading this gets the rules themselves to
    # reword, not the literal token {rules_block} it could only copy through.
    chain_module.SYSTEM_PROMPT = 'Judge against {rules_block}. Reply as {{"ok": true}}'
    _with_rules_block(chain_module)

    assert _model(chain_module, "rules_block").get_system_prompt() == (
        'Judge against RULE A\nRULE B. Reply as {"ok": true}'
    )


def test_a_resolver_that_needs_real_payload_data_falls_back_to_the_raw_prompt(chain_module):
    # generateSystemPrompt is called with an empty payload - prompt resolution
    # that depends on one entry's data belongs in insertInput() instead. One
    # that doesn't honour that must not take the run down.
    chain_module.SYSTEM_PROMPT = 'Judge {text}. Reply as {{"ok": true}}'

    def generateSystemPrompt(payload: dict) -> str:
        return chain_module.SYSTEM_PROMPT.format(text=payload["text"])

    chain_module.generateSystemPrompt = generateSystemPrompt

    assert _model(chain_module, "text").get_system_prompt() == (
        'Judge {text}. Reply as {"ok": true}'
    )


def test_a_resolver_that_returns_something_other_than_text_falls_back_too(chain_module):
    chain_module.SYSTEM_PROMPT = "Judge against {rules_block}"
    chain_module.generateSystemPrompt = lambda payload: ["not", "a", "prompt"]

    assert _model(chain_module, "rules_block").get_system_prompt() == (
        "Judge against {rules_block}"
    )


def test_a_rewrite_of_a_resolved_prompt_is_read_back_exactly_as_written(chain_module):
    # The modifier rewrites what it was shown (rules already written out), so
    # the rewrite has no {rules_block} left - a clean freeze. The next read
    # must still return that rewrite verbatim, braces and all.
    chain_module.SYSTEM_PROMPT = 'Judge against {rules_block}. Reply as {{"ok": true}}'
    _with_rules_block(chain_module)
    model = _model(chain_module, "rules_block")

    rewrite = 'Judge against RULE A (reworded)\nRULE B. Reply as {"ok": true}'
    model.set_new_system_prompt(rewrite)

    assert model.get_system_prompt() == rewrite


def test_a_rewrite_that_keeps_the_token_live_still_resolves_on_read(chain_module):
    chain_module.SYSTEM_PROMPT = "Judge against {rules_block}"
    _with_rules_block(chain_module)
    model = _model(chain_module, "rules_block")

    model.set_new_system_prompt("Judge strictly against {rules_block}")

    assert model.get_system_prompt() == "Judge strictly against RULE A\nRULE B"


def test_a_chain_with_no_prompt_of_its_own_says_which_chain_it_was(chain_module):

    with pytest.raises(AttributeError, match="template_checklist"):
        _model(chain_module).get_system_prompt()


def test_the_placeholders_come_from_the_chains_own_prompt(chain_module):
    chain_module.SYSTEM_PROMPT = "Rewrite {text} in {tone}"

    assert _model(chain_module)._detect_template_vars() == ("text", "tone")


def test_a_chain_with_no_prompt_declares_no_placeholders(chain_module):

    assert _model(chain_module)._detect_template_vars() == ()


def test_a_prompt_that_is_not_a_valid_template_ends_the_run(chain_module):

    chain_module.SYSTEM_PROMPT = "A stray { here, then {text}"

    with pytest.raises(SystemExit):
        _model(chain_module)._detect_template_vars()


def test_a_placeholder_the_resolver_fills_on_an_empty_payload_is_not_tracked(chain_module):
    # {rules_block} never needs a real corpus entry - generateSystemPrompt()
    # fills it from static config alone, so it's gone by the time the
    # modifier ever sees the prompt and there's nothing left to protect.
    chain_module.SYSTEM_PROMPT = "Judge against {rules_block}"
    _with_rules_block(chain_module)

    assert _model(chain_module)._detect_template_vars() == ()


def test_a_placeholder_the_resolver_cannot_fill_without_payload_is_still_tracked(chain_module):
    # generateSystemPrompt is called with an empty payload; one that reads
    # payload[...] raises on it and falls back to the raw, unresolved prompt
    # - so the placeholder it needed real data for is still there to protect.
    chain_module.SYSTEM_PROMPT = "Judge {text}"

    def generateSystemPrompt(payload: dict) -> str:
        return chain_module.SYSTEM_PROMPT.format(text=payload["text"])

    chain_module.generateSystemPrompt = generateSystemPrompt

    assert _model(chain_module)._detect_template_vars() == ("text",)


class _LeaveUnknownKeys(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def test_a_placeholder_the_resolver_leaves_unfilled_is_tracked_even_when_others_resolve(
    chain_module,
):
    # A resolver can fill some declared vars and deliberately leave others -
    # only the ones it actually resolved should stop being tracked.
    chain_module.SYSTEM_PROMPT = "Judge {text} against {rules_block}"

    def generateSystemPrompt(payload: dict) -> str:
        return chain_module.SYSTEM_PROMPT.format_map(_LeaveUnknownKeys(rules_block="RULE A"))

    chain_module.generateSystemPrompt = generateSystemPrompt

    assert _model(chain_module)._detect_template_vars() == ("text",)


def test_a_resolved_json_example_does_not_look_like_a_placeholder(chain_module):
    # Once rendered, {{"ok": true}} is single-braced literal text - it must
    # not be misread as a fresh field reference while checking what survived.
    chain_module.SYSTEM_PROMPT = 'Judge against {rules_block}. Reply as {{"ok": true}}'
    _with_rules_block(chain_module)

    assert _model(chain_module)._detect_template_vars() == ()


def test_the_detected_placeholders_are_what_callers_are_told(chain_module):

    model = _model(chain_module, "text")

    assert model.get_prompt_template_vars() == ("text",)


def test_a_rewrite_is_stored_as_a_format_template(chain_module):
    chain_module.SYSTEM_PROMPT = "Rewrite {text}."

    _model(chain_module, "text").set_new_system_prompt('Rewrite {text}. Reply as {"ok": true}')

    assert chain_module.SYSTEM_PROMPT == 'Rewrite {text}. Reply as {{"ok": true}}'


def test_the_runnable_is_rebuilt_so_the_new_prompt_takes_effect(chain_module):

    chain_module.SYSTEM_PROMPT = "Rewrite {text}."
    model = _model(chain_module, "text")

    model.set_new_system_prompt("A better prompt for {text}")

    assert model.rebuilds == ["rebuilt"]
    assert model.runnable == "a runnable"


def test_a_chain_with_no_prompt_cannot_be_rewritten(chain_module):
    with pytest.raises(AttributeError, match="template_checklist"):
        _model(chain_module).set_new_system_prompt("a new prompt")


def test_nothing_is_rebuilt_when_there_was_no_prompt_to_write(chain_module):

    model = _model(chain_module)

    with pytest.raises(AttributeError):
        model.set_new_system_prompt("a new prompt")

    assert model.rebuilds == []


@pytest.mark.parametrize(
    "template_vars, prompt",
    [
        (("text",), 'Rewrite {text}. Reply as {"ok": true}'),
        (("text",), "Braces } and { on their own, plus {text}"),
        (("score",), "Score {score:.2f}"),
        ((), 'Only literals: {"already": "doubled"}'),
        (("text",), "{text}"),
    ],
    ids=["json-example", "loose-braces", "format-spec", "no-fields", "bare-field"],
)
def test_a_prompt_written_then_read_is_the_prompt_that_was_written(
    chain_module, template_vars, prompt
):
    chain_module.SYSTEM_PROMPT = "whatever the chain shipped with"
    model = _model(chain_module, *template_vars)

    model.set_new_system_prompt(prompt)

    assert model.get_system_prompt() == prompt


@pytest.mark.parametrize(
    "prompt",
    [
        "Judge against { rules_block }",
        "Judge against {rules-block}",
    ],
    ids=["extra-spaces", "wrong-name"],
)
def test_a_rewrite_that_mangles_a_declared_placeholder_is_rejected(chain_module, prompt):
    chain_module.SYSTEM_PROMPT = "Judge against {rules_block}"
    model = _model(chain_module, "rules_block")

    with pytest.raises(ValueError, match="mangled placeholder"):
        model.set_new_system_prompt(prompt)

    assert model.rebuilds == []
    assert chain_module.SYSTEM_PROMPT == "Judge against {rules_block}"


def test_a_rewrite_that_cleanly_drops_a_declared_placeholder_freezes_it(chain_module):
    # No brace anywhere referencing rules_block - set_new_system_prompt()
    # only raises for a mangled attempt at keeping a placeholder live, not
    # for a clean drop (see ModifierModel._warn_if_placeholders_frozen for
    # where that gets flagged instead).
    chain_module.SYSTEM_PROMPT = "Judge against {rules_block}"
    model = _model(chain_module, "rules_block")

    model.set_new_system_prompt("Judge against the rules")

    assert model.rebuilds == ["rebuilt"]
    assert chain_module.SYSTEM_PROMPT == "Judge against the rules"


def test_a_clean_drop_of_one_placeholder_does_not_affect_another_thats_kept(chain_module):
    # missing={"rules_block"} while find_near_miss_placeholders() finds
    # nothing for it - the clean-drop path - must not disturb "text", which
    # is still present and kept live.
    chain_module.SYSTEM_PROMPT = "Judge {text} against {rules_block}"
    model = _model(chain_module, "text", "rules_block")

    model.set_new_system_prompt("Judge {text} strictly")

    assert model.rebuilds == ["rebuilt"]
    assert chain_module.SYSTEM_PROMPT == "Judge {text} strictly"


def test_a_clean_drop_is_not_confused_by_an_unrelated_brace_example(chain_module):
    # rules_block is cleanly dropped (no brace referencing it anywhere), but
    # the rewrite still contains an unrelated brace-delimited JSON example -
    # that must not be mistaken for a near-miss of rules_block.
    chain_module.SYSTEM_PROMPT = "Judge against {rules_block}"
    model = _model(chain_module, "rules_block")

    model.set_new_system_prompt('Judge against the rules. Reply as {"ok": true}')

    assert model.rebuilds == ["rebuilt"]
    assert chain_module.SYSTEM_PROMPT == 'Judge against the rules. Reply as {{"ok": true}}'


def test_a_mangled_placeholder_is_still_caught_alongside_a_clean_drop(chain_module):
    # Both are "missing" from the escaped text, but only "text" is a near
    # miss (mangled) - "rules_block" is a clean, intentional drop. Only the
    # mangled one may raise, and the raise must still block persistence.
    chain_module.SYSTEM_PROMPT = "Judge {text} against {rules_block}"
    model = _model(chain_module, "text", "rules_block")

    with pytest.raises(ValueError, match="mangled placeholder"):
        model.set_new_system_prompt("Judge { text } strictly")

    assert model.rebuilds == []
    assert chain_module.SYSTEM_PROMPT == "Judge {text} against {rules_block}"


def test_a_prompt_survives_being_rewritten_epoch_after_epoch(chain_module):

    chain_module.SYSTEM_PROMPT = 'Rewrite {text}. Reply as {{"ok": true}}'
    model = _model(chain_module, "text")

    for _ in range(5):
        model.set_new_system_prompt(model.get_system_prompt())

    assert model.get_system_prompt() == 'Rewrite {text}. Reply as {"ok": true}'
    assert chain_module.SYSTEM_PROMPT == 'Rewrite {text}. Reply as {{"ok": true}}'
