import json

from chain_checker.baseclasses.chain.lenient_json import escape_stray_quotes


def _roundtrip(text: str) -> object:
    return json.loads(escape_stray_quotes(text))


# --------------------------------------------------------------------------
# Valid JSON must survive completely untouched - the repair is a fallback,
# never a rewrite.
# --------------------------------------------------------------------------


def test_already_valid_json_comes_back_byte_identical():
    text = '{\n  "a": "x",\n  "b": [1, 2],\n  "c": {"d": true}\n}'
    assert escape_stray_quotes(text) == text


def test_a_legally_escaped_quote_is_not_escaped_a_second_time():
    text = r'{"a": "he said \"hi\""}'
    assert escape_stray_quotes(text) == text
    assert _roundtrip(text) == {"a": 'he said "hi"'}


def test_other_backslash_escapes_are_left_alone():
    text = r'{"a": "line\nbreak \\ and ä"}'
    assert escape_stray_quotes(text) == text


def test_structural_characters_inside_a_string_are_not_read_as_structure():
    text = '{"a": "}] , : {[", "b": true}'
    assert _roundtrip(text) == {"a": "}] , : {[", "b": True}


# --------------------------------------------------------------------------
# The actual repair: quotes the model forgot to escape inside a string.
# --------------------------------------------------------------------------


def test_a_quoted_word_mid_sentence_is_escaped():
    assert _roundtrip('{"evidence": "the word "ACME" is uppercase"}') == {
        "evidence": 'the word "ACME" is uppercase'
    }


def test_a_string_starting_with_a_quoted_word_is_escaped():
    assert _roundtrip('{"evidence": ""Early Bird" is hyphenated"}') == {
        "evidence": '"Early Bird" is hyphenated'
    }


def test_a_stray_quote_followed_by_a_comma_is_still_escaped():
    # The hard case: the stray quote after ACME is followed by a comma,
    # exactly like a real terminator would be. Only the last quote is
    # followed by a comma *and* the next "key": pair.
    reply = '{"evidence": "correct: "100% ACME", "ACME Partner" - fine.", "passed": true}'
    assert _roundtrip(reply) == {
        "evidence": 'correct: "100% ACME", "ACME Partner" - fine.',
        "passed": True,
    }


def test_a_stray_quote_before_the_closing_brace_is_escaped():
    assert _roundtrip('{"summary": "it reads as "marketing"."}') == {
        "summary": 'it reads as "marketing".'
    }


def test_keys_are_repaired_against_their_colon_rather_than_a_comma():
    assert _roundtrip('{"the "odd" key": 1, "b": 2}') == {'the "odd" key': 1, "b": 2}


def test_strings_in_an_array_still_separate_on_their_commas():
    assert _roundtrip('{"a": ["one", "two", "three"]}') == {"a": ["one", "two", "three"]}


def test_a_stray_quote_inside_an_array_element_is_escaped():
    assert _roundtrip('{"a": ["say "hi" now", "two"]}') == {"a": ['say "hi" now', "two"]}


def test_nested_objects_keep_track_of_which_container_they_are_in():
    reply = '{"items": [{"note": "a "quoted" word", "ok": true}], "n": 1}'
    assert _roundtrip(reply) == {"items": [{"note": 'a "quoted" word', "ok": True}], "n": 1}


# --------------------------------------------------------------------------
# Nothing here may turn a broken reply into an exception of its own: a
# document it cannot fix must simply come back unfixed, for the real parser
# to reject.
# --------------------------------------------------------------------------


def test_an_unterminated_string_is_returned_rather_than_raising():
    assert escape_stray_quotes('{"a": "never closed') == '{"a": "never closed'


def test_plain_prose_is_returned_unchanged():
    assert escape_stray_quotes("not json at all") == "not json at all"


def test_an_empty_reply_is_returned_unchanged():
    assert escape_stray_quotes("") == ""


def test_the_repair_only_ever_inserts_backslashes():
    # Guards the property the whole approach rests on: no character of the
    # model's answer can go missing, whatever the scanner decides.
    reply = '{"a": "one "two" three", "b": ["x", "y "z""], "c": 1}'
    repaired = escape_stray_quotes(reply)

    assert repaired.replace('\\"', '"') == reply
    assert len(repaired) > len(reply)


def test_the_real_world_checklist_reply_parses_with_every_inner_quote_kept():
    # Condensed from a real ollama reply that chain_checker dropped as an
    # empty prediction (case t-1787208637664281646): the answer was correct
    # in substance, only its quoting was broken.
    reply = """{
  "items": [
    {
      "rule_id": "brand-uppercase",
      "evidence": "All correct: "100% ACME", "ACME Partner" - every occurrence is uppercase.",
      "passed": true
    },
    {
      "rule_id": "seamless-tone",
      "evidence": ""Na zur Gutschrift" is garbled - should be "Jetzt zur Gutschrift".",
      "passed": false
    }
  ],
  "summary": "Correct brand usage, but fails on seamless tone (garbled "Na zur Gutschrift")."
}"""
    parsed = _roundtrip(reply)

    assert parsed == {
        "items": [
            {
                "rule_id": "brand-uppercase",
                "evidence": 'All correct: "100% ACME", "ACME Partner" '
                "- every occurrence is uppercase.",
                "passed": True,
            },
            {
                "rule_id": "seamless-tone",
                "evidence": '"Na zur Gutschrift" is garbled - should be "Jetzt zur Gutschrift".',
                "passed": False,
            },
        ],
        "summary": 'Correct brand usage, but fails on seamless tone (garbled "Na zur Gutschrift").',
    }
