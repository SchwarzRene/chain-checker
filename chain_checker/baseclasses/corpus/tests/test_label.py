import pytest

from chain_checker.baseclasses.corpus.label import (
    Label,
    describe_value,
    is_number,
    kind_conflict_message,
    value_kind,
)


def test_get_returns_the_stored_data():
    label = Label({"passed": True, "score": 0.9})

    assert label.get() == {"passed": True, "score": 0.9}


def test_get_keys_and_get_value():
    label = Label({"passed": True, "score": 0.9})

    assert label.get_keys() == ["passed", "score"]
    assert label.get_value("passed") is True
    assert label.get_value("score") == 0.9


def test_constructor_defensively_copies_the_input_dict():
    source = {"passed": True}
    label = Label(source)

    source["passed"] = False

    assert label.get_value("passed") is True


def test_get_returns_a_copy_not_a_reference():
    label = Label({"passed": True})

    returned = label.get()
    returned["passed"] = False

    assert label.get_value("passed") is True


def test_set_replaces_the_data_and_also_defensively_copies():
    label = Label({"passed": True})
    replacement = {"passed": False}

    label.set(replacement)
    replacement["passed"] = True

    assert label.get_value("passed") is False


def test_matches_reports_true_for_every_key_that_agrees():
    true_label = Label({"a": True, "b": 1})
    predicted = Label({"a": True, "b": 1})

    assert true_label.matches(predicted) == {"a": True, "b": True}


def test_matches_reports_false_for_a_disagreeing_key():
    true_label = Label({"a": True})
    predicted = Label({"a": False})

    assert true_label.matches(predicted) == {"a": False}


def test_matches_only_checks_this_labels_own_keys():
    true_label = Label({"a": True})
    predicted = Label({"a": True, "b": "extra, not checked"})

    assert true_label.matches(predicted) == {"a": True}


def test_matches_counts_a_key_missing_on_the_other_side_as_a_mismatch():
    true_label = Label({"a": True})
    predicted = Label({})

    assert true_label.matches(predicted) == {"a": False}


def test_matches_on_an_empty_label_is_vacuously_true():
    # An empty label has nothing to check, so all({}.values()) makes this
    # `True` by definition rather than by accident - Corpus rejects an empty
    # output at load time precisely so a case can never reach this trivially.
    empty_label = Label({})
    predicted = Label({"a": True})

    matches = empty_label.matches(predicted)

    assert matches == {}
    assert all(matches.values())


# --------------------------------------------------------------------------
# is_number / value_kind / describe_value: shared between Corpus (which needs
# to reject a key that disagrees on kind before a model call happens) and the
# metrics (which need the same kind to route a key to the right sub-metric).
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (3, True),
        (3.5, True),
        (-0.5, True),
        (True, False),
        (False, False),
        ("0.5", False),
        (None, False),
        ([1], False),
        ({}, False),
    ],
    ids=["int", "float", "negative", "true", "false", "numeric-string", "none", "list", "dict"],
)
def test_is_number_admits_what_can_be_subtracted_and_nothing_else(value, expected):
    # bool must be excluded even though it is an int subclass: True - 1 would
    # type-check but the result is meaningless for a boolean-valued key.
    assert is_number(value) is expected


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        (True, "bool"),
        (False, "bool"),
        (3, "number"),
        (3.5, "number"),
        ({"r1": True, "r2": False}, "bools"),
        # An empty mapping asserts "nothing to check" as a single verdict,
        # which is not the same shape as a populated dict-of-bool.
        ({}, "value"),
        ({"r1": 0.5}, "value"),
        ({"r1": True, "r2": 1}, "value"),
        ("en", "value"),
        (None, "value"),
        ([1, 2], "value"),
    ],
    ids=[
        "true",
        "false",
        "int",
        "float",
        "dict-of-bools",
        "empty-dict",
        "dict-of-floats",
        "dict-mixed",
        "string",
        "null",
        "list",
    ],
)
def test_value_kind_names_how_a_value_has_to_be_scored(value, kind):
    assert value_kind(value) == kind


def test_bool_is_kinded_before_number():
    # bool subclasses int, so the bool check has to run first or every
    # boolean value would be misread as a number.
    assert value_kind(True) == "bool"
    assert value_kind(1) == "number"


def test_an_int_and_a_float_are_one_kind():
    # Lets a corpus mix 3 and 3.5 on the same key without tripping the
    # kind-consistency check.
    assert value_kind(3) == value_kind(3.5)


def test_describe_value_shows_short_values_whole():
    assert describe_value({}) == "{}"
    assert describe_value(0.9) == "0.9"
    assert describe_value("en") == "'en'"


def test_describe_value_truncates_a_long_one():
    # A checked value can be an entire document, and this text is meant to
    # sit inline in a one-line error message.
    described = describe_value("x" * 200)

    assert len(described) == 40
    assert described.endswith("...")


# --------------------------------------------------------------------------
# kind_conflict_message
# --------------------------------------------------------------------------


def test_the_conflict_message_names_both_cases_and_both_values():
    # Whoever edits the corpus needs to see which two cases disagree and what
    # each one wrote, not just that a conflict exists.
    message = kind_conflict_message("score", "case-0", 0.9, "case-1", "high")

    assert "'case-0'" in message
    assert "'case-1'" in message
    assert "0.9" in message
    assert "'high'" in message
    assert "changes type between cases" in message


def test_the_conflict_message_names_the_file_when_there_is_one():
    # Corpus.load knows the .yaml path it read; a corpus assembled directly in
    # memory (as the metrics backstop does) has no path to report.
    with_path = kind_conflict_message("score", "a", 1, "b", "x", "corpus.yaml")
    without_path = kind_conflict_message("score", "a", 1, "b", "x")

    assert with_path.startswith("Corpus file 'corpus.yaml': output key 'score'")
    assert without_path.startswith("Corpus output key 'score'")


def test_the_conflict_message_is_one_text_either_way():
    # Both callers route through this one builder so the wording of the
    # rejection cannot drift between them - only the path prefix differs.
    with_path = kind_conflict_message("score", "a", 1, "b", "x", "corpus.yaml")
    without_path = kind_conflict_message("score", "a", 1, "b", "x")

    shared = "key 'score' changes type between cases"
    assert with_path[with_path.index(shared) :] == without_path[without_path.index(shared) :]


def test_the_conflict_message_truncates_a_long_value():
    # Goes through describe_value like everything else here, so a checked
    # document does not get dumped whole into the error text.
    message = kind_conflict_message("doc", "a", "x" * 200, "b", True)

    assert "x" * 200 not in message
    assert "..." in message
