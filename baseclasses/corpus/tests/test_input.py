# Input adds no behavior of its own over Record, so this only confirms
# Record's copy-in/copy-out contract still holds through the subclass.

from chain_checker.baseclasses.corpus.input import Input


def test_get_returns_the_stored_data():
    inp = Input({"text": "hello", "threshold": 0.8})

    assert inp.get() == {"text": "hello", "threshold": 0.8}


def test_get_keys_and_get_value():
    inp = Input({"text": "hello", "threshold": 0.8})

    assert inp.get_keys() == ["text", "threshold"]
    assert inp.get_value("text") == "hello"
    assert inp.get_value("threshold") == 0.8


def test_constructor_defensively_copies_the_input_dict():
    source = {"text": "hello"}
    inp = Input(source)

    source["text"] = "mutated after construction"

    assert inp.get_value("text") == "hello"


def test_get_returns_a_copy_not_a_reference():
    inp = Input({"text": "hello"})

    returned = inp.get()
    returned["text"] = "mutated via the getter"

    assert inp.get_value("text") == "hello"
