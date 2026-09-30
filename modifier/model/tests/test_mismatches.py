from chain_checker.modifier.model.mismatches import (
    diff_mismatches,
    simplify_mispredictions,
)


def test_matching_fields_produce_no_diff():
    assert diff_mismatches({"passed": True}, {"passed": True}) == {}


def test_a_disagreeing_top_level_field_is_reported():
    assert diff_mismatches({"passed": True}, {"passed": False}) == {
        "passed": {"true": True, "predicted": False},
    }


def test_a_field_the_prediction_never_produced_is_still_a_mismatch():
    assert diff_mismatches({"passed": True}, {}) == {
        "passed": {"true": True, "predicted": None},
    }


def test_extra_fields_the_prediction_volunteered_are_never_reported():
    assert diff_mismatches({"passed": True}, {"passed": True, "extra": "noise"}) == {}


def test_a_nested_dicts_disagreeing_subkey_is_flattened_to_dotted_notation():
    assert diff_mismatches(
        {"verdicts": {"r1": True, "r2": False}},
        {"verdicts": {"r1": True, "r2": True}},
    ) == {"verdicts.r2": {"true": False, "predicted": True}}


def test_a_nested_dicts_missing_subkey_is_still_a_mismatch():
    assert diff_mismatches({"verdicts": {"r1": True}}, {"verdicts": {}}) == {
        "verdicts.r1": {"true": True, "predicted": None},
    }


def test_a_nested_field_predicted_as_something_other_than_a_dict_fails_every_subkey():
    assert diff_mismatches({"verdicts": {"r1": True}}, {"verdicts": "not-a-dict"}) == {
        "verdicts.r1": {"true": True, "predicted": None},
    }


def test_an_empty_mapping_matching_an_empty_mapping_is_not_a_mismatch():
    assert diff_mismatches({"verdicts": {}}, {"verdicts": {}}) == {}


def test_simplify_mispredictions_keeps_the_input_and_reduces_the_diff():
    entries = {
        "case-1": {
            "input": {"tone": "friendly"},
            "true": {"passed": True},
            "predicted": {"passed": False},
        },
    }

    assert simplify_mispredictions(entries) == {
        "case-1": {
            "input": {"tone": "friendly"},
            "mismatches": {"passed": {"true": True, "predicted": False}},
        },
    }


def test_simplify_mispredictions_defaults_a_missing_input_to_empty():
    entries = {"case-1": {"true": {"passed": True}, "predicted": {"passed": True}}}

    assert simplify_mispredictions(entries)["case-1"]["input"] == {}


def test_simplify_mispredictions_of_no_entries_is_empty():
    assert simplify_mispredictions({}) == {}


def test_simplify_mispredictions_tolerates_a_missing_true_or_predicted_side():
    entries = {"case-1": {"input": {}, "true": None, "predicted": None}}

    assert simplify_mispredictions(entries) == {"case-1": {"input": {}, "mismatches": {}}}
