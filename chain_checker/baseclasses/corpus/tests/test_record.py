# Tested directly rather than only through Input/Label, so a regression in
# this shared base is caught here instead of being misread as a subclass bug.

from chain_checker.baseclasses.corpus.record import Record


def test_get_returns_the_stored_data():
    record = Record({"text": "hello", "threshold": 0.8})

    assert record.get() == {"text": "hello", "threshold": 0.8}


def test_get_keys_and_get_value():
    record = Record({"text": "hello", "threshold": 0.8})

    assert record.get_keys() == ["text", "threshold"]
    assert record.get_value("text") == "hello"
    assert record.get_value("threshold") == 0.8


def test_get_value_with_a_default_reads_an_optional_field():
    record = Record({"text": "hello"})

    assert record.get_value("threshold", 0.5) == 0.5
    assert record.get_value("text", "fallback") == "hello"


def test_a_default_of_none_is_returned_rather_than_raising():
    # A caller may legitimately want None back for an absent field, so
    # passing None as the default has to be distinguishable from passing no
    # default at all.
    record = Record({"text": "hello"})

    assert record.get_value("threshold", None) is None


def test_a_stored_none_is_returned_rather_than_the_default():
    # The field is present with a None value, so the default must not kick in.
    record = Record({"threshold": None})

    assert record.get_value("threshold", 0.5) is None


def test_constructor_defensively_copies_the_input_dict():
    source = {"text": "hello"}
    record = Record(source)

    source["text"] = "mutated after construction"

    assert record.get_value("text") == "hello"


def test_get_returns_a_copy_not_a_reference():
    record = Record({"text": "hello"})

    returned = record.get()
    returned["text"] = "mutated via the getter"

    assert record.get_value("text") == "hello"


def test_str_includes_the_class_name_and_every_field():
    record = Record({"text": "hello"})

    rendered = str(record)

    assert "Record:" in rendered
    assert "text: hello" in rendered
