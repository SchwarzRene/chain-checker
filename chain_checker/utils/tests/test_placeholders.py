import pytest

from chain_checker.utils.placeholders import (
    detect_placeholders,
    escape_stray_braces,
    field_root,
    find_near_miss_placeholders,
    unescape_braces,
)


@pytest.mark.parametrize(
    "prompt, expected",
    [
        ("Rewrite {text} now.", ("text",)),
        ("{a} then {b} then {c}", ("a", "b", "c")),
        ("{text} and {text} again", ("text",)),
        ("No fields at all.", ()),
        ("", ()),
        ("Return {{}} verbatim.", ()),
        ('Return {{"ok": true}} for {text}', ("text",)),
        ("{{{text}}}", ("text",)),
        ("{0} is positional, {text} is not", ("text",)),
    ],
    ids=[
        "one-field",
        "several-in-order",
        "duplicates-collapse",
        "no-fields",
        "empty-prompt",
        "escaped-pair-alone",
        "escaped-json-beside-a-field",
        "field-wrapped-in-escaped-braces",
        "positional-index-is-not-a-name",
    ],
)
def test_the_fields_a_prompt_declares_are_found(prompt, expected):
    assert detect_placeholders(prompt) == expected


@pytest.mark.parametrize(
    "prompt, expected",
    [
        ("Score {score:.2f}", ("score",)),
        ("Hello {user.name}", ("user",)),
        ("Item {items[0]}", ("items",)),
        ("Raw {text!r}", ("text",)),
    ],
    ids=["format-spec", "attribute", "index", "conversion"],
)
def test_a_field_carrying_a_suffix_still_names_the_variable_it_needs(prompt, expected):
    assert detect_placeholders(prompt) == expected


@pytest.mark.parametrize(
    "prompt",
    ["A stray { here, then {text}", "{text} first, then a stray {", "A lone } here"],
    ids=["leading", "trailing", "closing"],
)
def test_a_prompt_that_is_not_a_valid_template_is_refused(prompt):
    with pytest.raises(ValueError):
        detect_placeholders(prompt)


@pytest.mark.parametrize(
    "field_name, expected",
    [
        ("text", "text"),
        ("score:.2f", "score"),
        ("user.name", "user"),
        ("items[0]", "items"),
        ("text!r", "text"),
        ("_private", "_private"),
        ("0", None),
        ("", None),
        ('"ok": true', None),
    ],
    ids=[
        "bare",
        "spec",
        "attribute",
        "index",
        "conversion",
        "leading-underscore",
        "positional",
        "empty",
        "json-fragment",
    ],
)
def test_the_name_a_caller_has_to_supply_is_read_off_the_field(field_name, expected):
    assert field_root(field_name) == expected


@pytest.mark.parametrize(
    "stored, expected",
    [
        ("Rewrite {text}.", "Rewrite {text}."),
        ('{{"ok": true}}', '{"ok": true}'),
        ("{text} and {{literal}}", "{text} and {literal}"),
        ("{{{{doubly}}}}", "{{doubly}}"),
        ("no braces here", "no braces here"),
        ("", ""),
    ],
    ids=["field-only", "escaped-json", "mixed", "twice-escaped", "none", "empty"],
)
def test_a_stored_template_reads_back_as_the_text_a_model_sees(stored, expected):
    assert unescape_braces(stored) == expected


def test_a_field_followed_by_an_escaped_brace_keeps_both():
    assert unescape_braces("{text}}}") == "{text}}"


def test_a_declared_placeholder_is_left_alone():
    assert escape_stray_braces("Rewrite {text}.", ("text",)) == "Rewrite {text}."


def test_a_stray_brace_is_doubled():
    assert escape_stray_braces('{"ok": true}', ()) == '{{"ok": true}}'


@pytest.mark.parametrize(
    "template_vars, prompt",
    [
        (("text",), 'Rewrite {text}. Reply as {"ok": true}'),
        (("text",), "Rewrite {text}. Use } and { loosely."),
        (("a", "b"), "{a} before {b}, and a literal {brace}"),
        ((), 'No fields here, just {"json": [1, 2]}'),
        (("text",), "Nothing stray at all: {text}"),
    ],
    ids=["json-example", "unbalanced", "two-fields", "no-fields", "clean"],
)
def test_the_escaped_prompt_survives_the_format_call_it_is_built_for(template_vars, prompt):
    escaped = escape_stray_braces(prompt, template_vars)

    rendered = escaped.format(**{var: f"<{var}>" for var in template_vars})

    expected = prompt
    for var in template_vars:
        expected = expected.replace(f"{{{var}}}", f"<{var}>")
    assert rendered == expected


def test_an_undeclared_field_reaches_the_model_as_text_not_a_substitution():
    escaped = escape_stray_braces("Use {tone} for {text}", ("text",))

    assert escaped.format(text="hi") == "Use {tone} for hi"


def test_a_chain_with_no_declared_fields_escapes_everything():
    prompt = "Return {a} and {b}"

    assert escape_stray_braces(prompt, ()).format() == prompt


def test_a_declared_field_keeps_its_format_spec():
    escaped = escape_stray_braces("Score {score:.2f} today", ("score",))

    assert escaped.format(score=0.5) == "Score 0.50 today"


def test_a_declared_fields_attribute_and_index_survive_too():
    escaped = escape_stray_braces("{user.name} picked {items[0]}", ("user", "items"))

    assert escaped == "{user.name} picked {items[0]}"


def test_an_opening_brace_that_never_closes_is_escaped():
    assert escape_stray_braces("A stray { and then text", ("text",)) == ("A stray {{ and then text")


@pytest.mark.parametrize(
    "template_vars, prompt",
    [
        (("text",), 'Rewrite {text}. Reply as {"ok": true}'),
        (("text",), "Braces } and { on their own, plus {text}"),
        (("score",), "Score {score:.2f}"),
        ((), 'Only literals: {{"already": "doubled"}}'),
        (("text",), "{text}"),
    ],
    ids=["json-example", "loose-braces", "format-spec", "pre-doubled", "bare-field"],
)
def test_unescaping_undoes_escaping_exactly(template_vars, prompt):
    assert unescape_braces(escape_stray_braces(prompt, template_vars)) == prompt


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Judge against { rules_block }", ("rules_block",)),
        ("Judge against {rules-block}", ("rules_block",)),
        ("Judge against the rules", ()),
        ("Judge against {rules_block}", ("rules_block",)),
        ("{ a } and {b-c}", ("a", "b_c")),
        ("A stray { with no close", ()),
    ],
    ids=[
        "extra-spaces",
        "wrong-name",
        "clean-drop",
        "exact-match",
        "several",
        "unterminated",
    ],
)
def test_a_broken_brace_referencing_a_declared_var_is_a_near_miss(text, expected):
    assert find_near_miss_placeholders(text, ("rules_block", "a", "b_c")) == expected


def test_the_placeholders_a_prompt_declares_survive_being_escaped():
    prompt = 'Rewrite {text} in {tone}. Reply as {"ok": true}'

    escaped = escape_stray_braces(prompt, detect_placeholders(prompt))

    assert escaped.format(text="hi", tone="warm") == ('Rewrite hi in warm. Reply as {"ok": true}')
