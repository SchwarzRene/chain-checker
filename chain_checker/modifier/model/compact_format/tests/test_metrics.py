from chain_checker.modifier.model.compact_format.metrics import (
    format_accuracy_report,
    format_metric_block,
)


def test_a_boolean_rules_confusion_counts_are_rendered_as_a_table():
    block = format_metric_block(
        "Rule-Metrics",
        {
            "no-exclamation-spam": {"accuracy": 0.5, "TT": 1, "TF": 1, "FT": 0, "FF": 0},
        },
    )

    assert "## Rule-Metrics" in block
    assert "field | accuracy | TT | TF | FT | FF" in block
    assert "no-exclamation-spam | 0.5 | 1 | 1 | 0 | 0" in block


def test_several_boolean_rules_share_one_table():
    block = format_metric_block(
        "Rule-Metrics",
        {
            "rule-a": {"accuracy": 1.0, "TT": 2, "TF": 0, "FT": 0, "FF": 0},
            "rule-b": {"accuracy": 0.0, "TT": 0, "TF": 0, "FT": 2, "FF": 0},
        },
    )

    assert block.count("field | accuracy | TT | TF | FT | FF") == 1
    assert "rule-a | 1 |" in block
    assert "rule-b | 0 |" in block


def test_a_numeric_rules_mae_is_rendered_as_one_line_not_a_table():
    block = format_metric_block(
        "Rule-Metrics",
        {
            "room_count": {"mae": 1.5, "true_scores": [2, 4], "predicted_scores": [1, 5]},
        },
    )

    assert "room_count: mae=1.5" in block
    assert "true_scores=[2, 4]" in block
    assert "predicted_scores=[1, 5]" in block
    assert "TT" not in block


def test_a_plain_scalar_metric_is_one_key_value_line():
    assert format_metric_block("Token-Metrics", {"total_tokens": 42}) == (
        "## Token-Metrics\ntotal_tokens: 42"
    )


def test_a_metric_reported_as_a_bare_list_is_rendered_inline():
    assert format_metric_block("Something", [1, 2, 3]) == "## Something\n[1, 2, 3]"


def test_a_metric_reported_as_a_bare_scalar_is_rendered_inline():
    assert format_metric_block("Something", 7) == "## Something\n7"


def test_a_dict_value_that_is_not_a_recognised_metric_shape_falls_back_to_repr():
    block = format_metric_block("Weird-Metrics", {"detail": {"nested": "value"}})

    assert "detail: {'nested': 'value'}" in block


def test_format_accuracy_report_joins_every_items_block_with_a_blank_line():
    report = format_accuracy_report(
        [
            {"name": "A", "results": {"total": 1}},
            {"name": "B", "results": {"total": 2}},
        ]
    )

    assert report == "## A\ntotal: 1\n\n## B\ntotal: 2"


def test_format_accuracy_report_of_no_items_is_an_empty_string():
    # Not every epoch necessarily has a scoreable metric; joining zero
    # blocks must not raise or print a stray header.
    assert format_accuracy_report([]) == ""


def test_nested_per_rule_accuracy_is_rendered_as_a_table_not_a_raw_dict():
    block = format_metric_block(
        "Accuracy-Metrics",
        {
            "accuracy": 0.6,
            "rules": {
                "verdicts": {
                    "sub_rules": {
                        "human-centered": {
                            "results": {"accuracy": 0.5, "TT": 8, "TF": 2, "FT": 13, "FF": 7}
                        }
                    }
                },
                "passed": {"results": {"accuracy": 0.9, "TT": 5, "TF": 1, "FT": 2, "FF": 22}},
            },
        },
    )

    assert "verdicts.human-centered | 0.5 | 8 | 2 | 13 | 7" in block
    assert "passed | 0.9 | 5 | 1 | 2 | 22" in block
    assert "sub_rules" not in block
    assert "accuracy: 0.6" in block
