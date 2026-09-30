from chain_checker.modifier.model.rule_trend import (
    format_rule_trend,
    rule_results_from_accuracy,
)


def _rule(accuracy: float, tf: int, ft: int) -> dict:
    return {"accuracy": accuracy, "TT": 10, "TF": tf, "FT": ft, "FF": 10}


def _row(report: str, rule: str) -> str:
    return next(line for line in report.splitlines() if line.startswith(f"{rule} |"))


def test_rule_results_are_flattened_to_dotted_names():
    accuracy_results = {
        "accuracy": 0.6,
        "rules": {"verdicts": {"sub_rules": {"hc": {"results": _rule(0.5, 1, 2)}}}},
    }

    assert rule_results_from_accuracy(accuracy_results) == {"verdicts.hc": _rule(0.5, 1, 2)}


def test_no_per_rule_results_gives_no_section():
    assert format_rule_trend([{}, {}], tolerance=0.02) == ""


def test_each_run_gets_a_column_with_accuracy_and_tf_ft():
    report = format_rule_trend([{"hc": _rule(0.667, 5, 5)}, {"hc": _rule(0.6, 3, 9)}], 0.02)

    assert "rule | Run-0 | Run-1 | best | trend" in report
    assert _row(report, "hc").startswith("hc | 0.667 5/5 | 0.600 3/9 | Run-0 |")


def test_a_rule_falling_every_run_is_flagged():
    history = [{"hc": _rule(a, 1, 1)} for a in (0.667, 0.6, 0.5, 0.467)]

    row = _row(format_rule_trend(history, 0.02), "hc")

    assert "FALLING 3 runs in a row" in row
    assert "below its best for 3 run(s)" in row


def test_a_single_dip_is_not_a_falling_trend():
    history = [{"hc": _rule(a, 1, 1)} for a in (0.6, 0.7, 0.65)]

    assert "FALLING" not in _row(format_rule_trend(history, 0.02), "hc")


def test_a_rising_dominant_error_is_named_with_its_counts():
    history = [{"hc": _rule(0.6, 2, ft)} for ft in (5, 9, 13)]

    row = _row(format_rule_trend(history, 0.02), "hc")

    assert "mostly FT (expected false, predicted true - judged true too easily)" in row
    assert "FT RISING 5 -> 9 -> 13" in row


def test_mostly_tf_is_named_as_judged_false_too_easily():
    row = _row(format_rule_trend([{"bu": _rule(0.6, 13, 0)}], 0.02), "bu")

    assert "mostly TF (expected true, predicted false - judged false too easily)" in row


def test_a_rule_missing_from_one_run_shows_a_dash_and_breaks_the_streak():
    history = [{"hc": _rule(0.8, 1, 1)}, {}, {"hc": _rule(0.7, 1, 1)}, {"hc": _rule(0.6, 1, 1)}]

    row = _row(format_rule_trend(history, 0.02), "hc")

    assert "| - |" in row
    assert "FALLING 1" not in row and "FALLING 2" not in row
