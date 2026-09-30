from chain_checker.modifier.model.parsing_report import build_parse_failure_report


def test_no_failed_ids_is_an_empty_report():
    assert build_parse_failure_report({"failed_ids": []}, {}) == {}


def test_a_failed_ids_input_is_borrowed_from_the_mispredicted_metrics_entry():
    parsing = {"failed_ids": ["case-1"]}
    mispredicted = {"case-1": {"input": {"tone": "friendly"}, "true": {}, "predicted": {}}}

    assert build_parse_failure_report(parsing, mispredicted) == {
        "case-1": {"input": {"tone": "friendly"}},
    }


def test_a_failed_id_with_no_matching_mispredicted_entry_gets_an_empty_input():
    assert build_parse_failure_report({"failed_ids": ["case-1"]}, {}) == {
        "case-1": {"input": {}},
    }


def test_ids_are_matched_as_strings_even_if_one_side_uses_an_int():
    parsing = {"failed_ids": [1]}
    mispredicted = {"1": {"input": {"a": 1}}}

    assert build_parse_failure_report(parsing, mispredicted) == {"1": {"input": {"a": 1}}}


def test_a_missing_failed_ids_key_is_treated_as_no_failures():
    assert build_parse_failure_report({}, {"case-1": {"input": {"a": 1}}}) == {}
