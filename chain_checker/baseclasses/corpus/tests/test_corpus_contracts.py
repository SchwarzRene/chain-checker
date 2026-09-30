"""Contract tests for the corpus API surface that no existing test reached.

Three groups: the lookup/iteration contracts on Corpus, the string and
missing-key contracts on Record and Entry, and the cross-layer comparison
that the whole ModelOutput-inherits-from-Label design exists to support -
Label.matches(ModelOutput) - which had no test at all despite being the
reason for the inheritance.
"""

import pytest

from chain_checker.baseclasses.corpus.corpus import Corpus
from chain_checker.baseclasses.corpus.entry import Entry
from chain_checker.baseclasses.corpus.input import Input
from chain_checker.baseclasses.corpus.label import Label
from chain_checker.baseclasses.corpus.output import ModelOutput
from chain_checker.baseclasses.corpus.output_empty import EmptyModelOutput
from chain_checker.baseclasses.corpus.record import Record


def _entry(entry_id, text: str = "hello") -> Entry:
    return Entry(
        entry_id,
        Input({"text": text}),
        Label({"passed": True}),
        {},
        EmptyModelOutput(),
    )


def _corpus(*ids) -> Corpus:
    corpus = Corpus()
    for entry_id in ids:
        corpus.add_entry(_entry(entry_id))
    return corpus


# --------------------------------------------------------------------------
# Corpus lookup contracts
# --------------------------------------------------------------------------


def test_get_by_id_raises_keyerror_for_an_unknown_id():
    # Each lookup raises the error its own kind of miss deserves - KeyError
    # here, IndexError from get_by_idx - and both carry a message naming what
    # the corpus actually holds rather than letting the bare container error
    # out.
    corpus = _corpus(1)

    with pytest.raises(KeyError, match="no entry with id"):
        corpus.get_by_id("no-such-id")


def test_get_by_idx_follows_plain_list_semantics_for_negative_indices():
    # A negative index counts from the end, and one past either end raises
    # rather than wrapping round onto an unrelated entry.
    corpus = _corpus("a", "b", "c")

    assert corpus.get_by_idx(-1).get_id() == "c"
    assert corpus.get_by_idx(-3).get_id() == "a"

    with pytest.raises(IndexError):
        corpus.get_by_idx(-4)

    with pytest.raises(IndexError):
        corpus.get_by_idx(3)


def test_get_by_idx_preserves_insertion_order():
    corpus = _corpus("first", "second", "third")

    assert [corpus.get_by_idx(i).get_id() for i in range(3)] == [
        "first",
        "second",
        "third",
    ]


def test_str_reports_the_entry_count():
    corpus = _corpus(1, 2, 3)

    rendered = str(corpus)

    assert "Corpus" in rendered
    assert "3" in rendered


def test_str_of_an_empty_corpus_reports_zero():
    assert "0" in str(Corpus())


# --------------------------------------------------------------------------
# Iteration contracts
# --------------------------------------------------------------------------


def test_each_call_to_iter_returns_an_independent_cursor():
    # Nested iteration over the same corpus must not share position - a
    # metric that compares every entry against every other entry needs this.
    corpus = _corpus(1, 2)

    pairs = [(outer.get_id(), inner.get_id()) for outer in corpus for inner in corpus]

    assert pairs == [(1, 1), (1, 2), (2, 1), (2, 2)]


def test_the_iterator_snapshots_the_entries_at_construction():
    # __iter__ returns iter(list(...)), copying the values up front, so
    # adding an entry mid-iteration neither appears in the running loop nor
    # raises the "dictionary changed size during iteration" RuntimeError a
    # bare dict iterator would. The list() is load-bearing: iterating
    # self._entries.values() directly would raise here.
    corpus = _corpus(1)
    iterator = iter(corpus)

    corpus.add_entry(_entry(2))
    seen = [entry.get_id() for entry in iterator]

    assert seen == [1]
    assert len(corpus) == 2


def test_iterating_an_empty_corpus_yields_nothing():
    assert list(Corpus()) == []


def test_reset_on_an_empty_corpus_is_a_no_op():
    corpus = Corpus()

    corpus.reset()

    assert len(corpus) == 0


def test_reset_replaces_every_prediction_not_just_the_first():
    corpus = _corpus(1, 2, 3)
    for entry in corpus:
        entry.get_model_output().set_output({"passed": True})

    corpus.reset()

    assert all(entry.get_model_output().get() == {} for entry in corpus)


# --------------------------------------------------------------------------
# Record / Entry contracts
# --------------------------------------------------------------------------


def test_get_value_raises_keyerror_for_a_missing_field():
    record = Record({"text": "hello"})

    with pytest.raises(KeyError):
        record.get_value("threshold")


def test_str_of_an_empty_record_is_just_the_class_name():
    assert str(Record({})) == "Record:\n"


def test_str_uses_the_subclass_name():
    assert str(Input({"text": "hello"})).startswith("Input:")
    assert str(Label({"passed": True})).startswith("Label:")


def test_a_subclass_can_override_the_formatting_hook():
    # __str__ dispatches through self._format_data, so an override takes
    # effect. It used to call Record._format_data by name, which made an
    # override look like it should work while silently doing nothing - the
    # class name in the same f-string was polymorphic, the formatter was not.
    class Loud(Record):
        @staticmethod
        def _format_data(data: dict) -> str:
            return "OVERRIDDEN"

    assert str(Loud({"a": 1})) == "Loud:\nOVERRIDDEN"


def test_entry_str_includes_the_id_and_every_component():
    entry = _entry(42, text="the input text")

    rendered = str(entry)

    assert "Entry 42" in rendered
    assert "the input text" in rendered
    assert "Input:" in rendered
    assert "Label:" in rendered


def test_get_info_returns_a_falsy_stored_value_rather_than_none():
    # get(key, None) cannot distinguish "absent" from "present and None",
    # but it must not swallow 0 / "" / False. Pinned because a caller
    # writing `if entry.get_info("retries"):` would misread a stored 0.
    entry = Entry(
        1,
        Input({}),
        Label({"passed": True}),
        {"retries": 0, "note": ""},
        EmptyModelOutput(),
    )

    assert entry.get_info("retries") == 0
    assert entry.get_info("note") == ""
    assert entry.get_info("absent") is None


def test_the_info_dict_is_defensively_copied_like_record_does():
    # Entry used to store the caller's mapping by reference, so anything
    # still holding it could rewrite a loaded entry's metadata after the
    # fact - inconsistent with the copying Record documents for Input/Label.
    info = {"language": "en"}
    entry = Entry(1, Input({}), Label({"passed": True}), info, EmptyModelOutput())

    info["language"] = "de"

    assert entry.get_info("language") == "en"


# --------------------------------------------------------------------------
# Cross-layer comparison: the reason ModelOutput inherits from Label
# --------------------------------------------------------------------------


def test_a_label_matches_a_model_output_that_agrees_on_every_checked_key():
    truth = Label({"passed": True, "score": 0.9})
    prediction = ModelOutput("some conversation", {"passed": True, "score": 0.9})

    assert truth.matches(prediction) == {"passed": True, "score": True}


def test_a_label_reports_the_specific_key_a_model_output_got_wrong():
    truth = Label({"passed": True, "score": 0.9})
    prediction = ModelOutput("", {"passed": True, "score": 0.1})

    assert truth.matches(prediction) == {"passed": True, "score": False}


def test_a_label_ignores_extra_keys_the_model_volunteered():
    truth = Label({"passed": True})
    prediction = ModelOutput("", {"passed": True, "reasoning": "because"})

    assert truth.matches(prediction) == {"passed": True}


def test_a_label_counts_every_key_as_wrong_against_an_empty_prediction():
    truth = Label({"passed": True, "score": 0.9})

    assert truth.matches(EmptyModelOutput()) == {"passed": False, "score": False}


def test_set_output_keeps_the_comparable_view_in_step_with_the_raw_output():
    # ModelOutput holds the same data twice - _output and Label's _data -
    # and set_output is what keeps them in step. A prediction mutated after
    # construction must compare on its new value, not its old one.
    truth = Label({"passed": True})
    prediction = EmptyModelOutput()
    assert truth.matches(prediction) == {"passed": False}

    prediction.set_output({"passed": True})

    assert truth.matches(prediction) == {"passed": True}
    assert prediction.get_output() == {"passed": True}


def test_matches_is_asymmetric():
    # matches() only reports on the receiver's own keys, so swapping the
    # operands answers a different question. Calling it on the prediction
    # instead of the truth silently grades the wrong direction.
    truth = Label({"passed": True})
    prediction = ModelOutput("", {"passed": True, "reasoning": "because"})

    assert truth.matches(prediction) == {"passed": True}
    assert prediction.matches(truth) == {"passed": True, "reasoning": False}


def test_a_bool_label_is_not_satisfied_by_an_int_prediction():
    # Python's True == 1 and False == 0, so plain == let a chain returning
    # the integer 1 pass a `passed: true` case and never caught a mistyped
    # output. bool and number are now decided separately.
    assert Label({"passed": True}).matches(Label({"passed": 1})) == {"passed": False}
    assert Label({"passed": False}).matches(Label({"passed": 0})) == {"passed": False}
    assert Label({"passed": True}).matches(Label({"passed": True})) == {"passed": True}
    assert Label({"passed": False}).matches(Label({"passed": False})) == {"passed": True}


def test_an_int_label_is_not_satisfied_by_a_bool_prediction():
    # The same guard in the other direction.
    assert Label({"n": 1}).matches(Label({"n": True})) == {"n": False}
    assert Label({"n": 0}).matches(Label({"n": False})) == {"n": False}


def test_floats_are_compared_with_a_tolerance():
    # 0.1 + 0.2 is 0.30000000000000004, so exact equality fails a value
    # correct to 16 digits. Float-valued output keys are ordinary here: a
    # score, a threshold, an error count.
    assert Label({"score": 0.3}).matches(Label({"score": 0.1 + 0.2})) == {"score": True}
    assert Label({"score": 1.0}).matches(Label({"score": 1})) == {"score": True}

    # isclose's rel_tol scales with magnitude, so at zero it needs an abs_tol
    # or it degenerates to exact equality - on the very cases the tolerance
    # exists for. An expected 0.0 is ordinary (a score, an error count).
    assert Label({"score": 0.0}).matches(Label({"score": 5.5e-17})) == {"score": True}
    assert Label({"score": 0.0}).matches(Label({"score": 1e-12})) == {"score": True}
    assert Label({"score": 0.0}).matches(Label({"score": -1e-12})) == {"score": True}


def test_a_genuinely_different_float_still_fails():
    # The tolerance must not swallow a real disagreement.
    assert Label({"score": 0.9}).matches(Label({"score": 0.1})) == {"score": False}
    # Guards abs_tol from being widened: 1e-6 is a real difference on a score.
    assert Label({"score": 0.0}).matches(Label({"score": 1e-6})) == {"score": False}


def test_a_float_label_is_not_satisfied_by_a_non_numeric_prediction():
    assert Label({"score": 0.5}).matches(Label({"score": "0.5"})) == {"score": False}
    assert Label({"score": 0.5}).matches(Label({"score": None})) == {"score": False}


def test_non_numeric_values_still_compare_by_equality():
    # Strings, lists and dicts are unaffected by the numeric special-casing.
    assert Label({"tag": "hello"}).matches(Label({"tag": "hello"})) == {"tag": True}
    assert Label({"tag": "hello"}).matches(Label({"tag": "world"})) == {"tag": False}
    assert Label({"v": {"a": 1}}).matches(Label({"v": {"a": 1}})) == {"v": True}
    assert Label({"v": [1, 2]}).matches(Label({"v": [1, 2]})) == {"v": True}


# --------------------------------------------------------------------------
# The bool and float rules apply at every depth, not just to a bare value
# --------------------------------------------------------------------------


def test_a_nested_bool_is_not_satisfied_by_a_nested_int():
    # `verdicts` is this codebase's canonical dict-of-bools output key, and it
    # used to fall through to `==` - so {"r1": True} == {"r1": 1} graded the
    # exact mistake the top-level guard was added to catch as a PASS. The same
    # output was graded two different ways depending only on its nesting.
    truth = Label({"verdicts": {"r1": True, "r2": False}})

    assert truth.matches(Label({"verdicts": {"r1": 1, "r2": False}})) == {"verdicts": False}
    assert truth.matches(Label({"verdicts": {"r1": True, "r2": 0}})) == {"verdicts": False}
    assert truth.matches(Label({"verdicts": {"r1": True, "r2": False}})) == {"verdicts": True}


def test_a_bool_inside_a_list_is_not_satisfied_by_an_int():
    truth = Label({"flags": [True, False]})

    assert truth.matches(Label({"flags": [1, False]})) == {"flags": False}
    assert truth.matches(Label({"flags": [True, False]})) == {"flags": True}


def test_nested_floats_get_the_same_tolerance_as_a_bare_one():
    # The mirror image: a list of scores correct to 16 digits used to report
    # FAIL, because only a top-level float reached math.isclose.
    assert Label({"scores": [0.3]}).matches(Label({"scores": [0.1 + 0.2]})) == {"scores": True}
    assert Label({"s": {"a": 0.3}}).matches(Label({"s": {"a": 0.1 + 0.2}})) == {"s": True}
    assert Label({"s": {"a": [0.3]}}).matches(Label({"s": {"a": [0.1 + 0.2]}})) == {"s": True}


def test_a_genuinely_different_nested_float_still_fails():
    assert Label({"scores": [0.9]}).matches(Label({"scores": [0.1]})) == {"scores": False}


def test_recursion_does_not_loosen_the_shape_a_value_has_to_have():
    # Only how a leaf is compared changed. A nested dict still needs the same
    # key set and a list the same length and order, exactly as `==` required -
    # matches() ignoring unasked-for keys is a property of the top-level
    # output block, not of a value the corpus spelled out in full.
    truth = Label({"v": {"a": True}})

    assert truth.matches(Label({"v": {"a": True, "b": True}})) == {"v": False}
    assert truth.matches(Label({"v": {}})) == {"v": False}
    assert Label({"v": [1, 2]}).matches(Label({"v": [1, 2, 3]})) == {"v": False}
    assert Label({"v": [1, 2]}).matches(Label({"v": [2, 1]})) == {"v": False}


def test_a_container_is_not_satisfied_by_a_different_container_type():
    assert Label({"v": {"a": 1}}).matches(Label({"v": [1]})) == {"v": False}
    assert Label({"v": [1]}).matches(Label({"v": {"a": 1}})) == {"v": False}
    assert Label({"v": [1, 2]}).matches(Label({"v": (1, 2)})) == {"v": False}


def test_recursion_reaches_arbitrarily_deep():
    truth = Label({"a": {"b": {"c": [{"d": True}]}}})

    assert truth.matches(Label({"a": {"b": {"c": [{"d": True}]}}})) == {"a": True}
    assert truth.matches(Label({"a": {"b": {"c": [{"d": 1}]}}})) == {"a": False}


def test_matches_grades_against_any_record():
    # matches() only calls other.get(), and its annotation now says Record
    # rather than Label, so an Input is a legitimate argument.
    assert Label({"text": "hello"}).matches(Input({"text": "hello"})) == {"text": True}
