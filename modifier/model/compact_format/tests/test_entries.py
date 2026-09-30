from chain_checker.modifier.model.compact_format.entries import (
    format_false_examples_report,
    format_parse_failure_report,
)


def test_no_entries_reports_that_nothing_was_mispredicted():
    assert format_false_examples_report({}) == (
        "(no mispredicted entries this run - every case already passed)"
    )


def test_no_entries_reports_that_nothing_failed_to_parse():
    assert format_parse_failure_report({}) == (
        "(no parsing failures this run - every entry produced a scoreable answer)"
    )


def test_a_short_scalar_input_field_stays_on_the_header_line():
    report = format_false_examples_report(
        {
            "case-1": {"input": {"tone": "friendly"}, "mismatches": {}},
        }
    )

    assert report == "### case-1 (tone=friendly)"


def test_a_long_input_field_is_pushed_into_its_own_body_block():
    long_text = "x" * 90
    report = format_false_examples_report(
        {
            "case-1": {"input": {"text": long_text}, "mismatches": {}},
        }
    )

    assert f"### case-1\n{long_text}" == report


def test_a_single_body_field_is_printed_without_its_key_as_a_label():
    long_text = "line one\nline two"
    report = format_false_examples_report(
        {
            "case-1": {"input": {"text": long_text}, "mismatches": {}},
        }
    )

    assert report == f"### case-1\n{long_text}"


def test_two_body_fields_are_each_labelled_by_key():
    body_a = "a" * 90
    body_b = "b" * 90
    report = format_false_examples_report(
        {
            "case-1": {"input": {"first": body_a, "second": body_b}, "mismatches": {}},
        }
    )

    assert f"first:\n{body_a}" in report
    assert f"second:\n{body_b}" in report


def test_mismatches_are_rendered_as_true_arrow_predicted():
    report = format_false_examples_report(
        {
            "case-1": {
                "input": {"tone": "friendly"},
                "mismatches": {"passed": {"true": True, "predicted": False}},
            },
        }
    )

    assert "-- mismatches --" in report
    assert "passed: true→false" in report


def test_an_entry_with_no_mismatches_key_prints_no_mismatches_section():
    # build_parse_failure_report()'s entries never carry a "mismatches" key
    # at all - .get() reading that as None must behave the same as {}.
    report = format_false_examples_report({"case-1": {"input": {}}})

    assert "-- mismatches --" not in report


def test_entries_sharing_the_same_input_print_it_once():
    entries = {
        "rule-a": {
            "input": {"text": "same text"},
            "mismatches": {"a": {"true": True, "predicted": False}},
        },
        "rule-b": {
            "input": {"text": "same text"},
            "mismatches": {"b": {"true": True, "predicted": False}},
        },
    }

    report = format_false_examples_report(entries)

    assert "### rule-a (text=same text)" in report
    assert "### rule-b\n(same input as rule-a)" in report
    assert report.count("text=same text") == 1


def test_entries_with_different_input_are_not_deduplicated():
    entries = {
        "case-1": {"input": {"text": "one"}, "mismatches": {}},
        "case-2": {"input": {"text": "two"}, "mismatches": {}},
    }

    report = format_false_examples_report(entries)

    assert "same input as" not in report


def test_entries_with_no_recorded_input_are_not_deduplicated():
    # "no input recorded" for both entries is not the same claim as "these
    # two entries share one input" - the two must never be conflated.
    entries = {
        "case-1": {"input": {}, "mismatches": {}},
        "case-2": {"input": {}, "mismatches": {}},
    }

    report = format_false_examples_report(entries)

    assert "same input as" not in report


def test_parse_failures_use_their_own_empty_and_entry_wording():
    report = format_parse_failure_report({"case-1": {"input": {"tone": "x"}}})

    assert report == "### case-1 (tone=x)"


def test_an_input_already_shown_in_an_earlier_run_is_only_referenced():
    report = format_false_examples_report(
        {
            "case-1": {
                "input": {"text": "x" * 90},
                "mismatches": {"verdicts.a": {"true": False, "predicted": True}},
            }
        },
        shown_in={"case-1": 2},
    )

    assert report.startswith("### case-1 (input: see Run-2)")
    assert "x" * 90 not in report
    # The diff is per run, so it is still shown in full.
    assert "verdicts.a: false→true" in report


def test_parse_failure_report_also_references_an_already_shown_input():
    report = format_parse_failure_report(
        {"case-1": {"input": {"text": "x" * 90}}}, shown_in={"case-1": 0}
    )

    assert report == "### case-1 (input: see Run-0)"
