# Entry only wires Input/Label/ModelOutput/info together; test_corpus_contracts.py
# covers the trickier edges (defensive copying, falsy info values).

from chain_checker.baseclasses.corpus.entry import Entry
from chain_checker.baseclasses.corpus.input import Input
from chain_checker.baseclasses.corpus.label import Label
from chain_checker.baseclasses.corpus.output_empty import EmptyModelOutput


def _make_entry(info: dict | None = None) -> Entry:
    return Entry(
        1,
        Input({"text": "hello"}),
        Label({"passed": True}),
        info if info is not None else {},
        EmptyModelOutput(),
    )


def test_getters_return_what_was_passed_in():
    entry = _make_entry()

    assert entry.get_id() == 1
    assert entry.get_input().get() == {"text": "hello"}
    assert entry.get_output().get() == {"passed": True}
    assert entry.get_model_output().get() == {}


def test_get_info_reads_from_the_free_form_dict():
    entry = _make_entry(info={"language": "en", "labeller": "alice"})

    assert entry.get_info("language") == "en"
    assert entry.get_info("labeller") == "alice"


def test_get_info_returns_none_for_a_missing_key():
    entry = _make_entry(info={"language": "en"})

    assert entry.get_info("nonexistent") is None


def test_set_model_output_replaces_the_prediction():
    entry = _make_entry()
    new_prediction = EmptyModelOutput()

    entry.set_model_output(new_prediction)

    assert entry.get_model_output() is new_prediction
