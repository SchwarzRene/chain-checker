"""The RULE-TREND section of the modifier's instruction: one row per checked
rule across every run so far, so a rule that gets worse on each rewrite is
flagged as such rather than left for the modifier to spot by comparing
several ACCURACY-REPORTs by eye."""

from typing import Any

from chain_checker.baseclasses.metrics.m_single_value import iter_rule_results

RuleResults = dict[str, dict[str, Any]]

# The streak of falls or rises that is reported as a trend rather than noise.
_MIN_STREAK = 2


def rule_results_from_accuracy(accuracy_results: dict[str, Any]) -> RuleResults:
    """{"verdicts.human-centered": {"accuracy", "TT", "TF", ...}} for every
    rule of one run's Accuracy-Metrics results that reports an accuracy."""
    return {
        ".".join(path): results
        for path, results in iter_rule_results(accuracy_results)
        if isinstance(results.get("accuracy"), (int, float))
    }


def _accuracy_series(history: list[RuleResults], rule: str) -> list[float | None]:
    return [run[rule]["accuracy"] if rule in run else None for run in history]


def _trailing_streak(values: list[int | float | None], rising: bool) -> int:
    """How many run-to-run steps in a row, ending at the latest run, moved
    in the given direction. A missing value ends the streak."""
    streak = 0
    for later, earlier in zip(reversed(values), list(reversed(values))[1:]):
        if later is None or earlier is None:
            break
        if (later > earlier) if rising else (later < earlier):
            streak += 1
        else:
            break
    return streak


def _format_cell(results: dict[str, Any] | None) -> str:
    if results is None:
        return "-"
    cell = f"{results['accuracy']:.3f}"
    if "TF" in results and "FT" in results:
        cell += f" {results['TF']}/{results['FT']}"
    return cell


def _best_run(series: list[float | None]) -> int:
    scored = [(i, a) for i, a in enumerate(series) if a is not None]
    return max(scored, key=lambda pair: pair[1])[0]


def best_rule_runs(history: list[RuleResults]) -> set[int]:
    """The run at which each rule scored best - the runs whose wording a
    rewrite may want to restore for that one rule."""
    rules = dict.fromkeys(rule for run in history for rule in run)
    return {_best_run(_accuracy_series(history, rule)) for rule in rules}


def _direction_flag(history: list[RuleResults], rule: str) -> str | None:
    latest = history[-1].get(rule)
    if latest is None or "TF" not in latest or "FT" not in latest:
        return None

    tf, ft = latest["TF"], latest["FT"]
    if ft > tf and ft >= 2:
        key, meaning = "FT", "expected false, predicted true - judged true too easily"
    elif tf > ft and tf >= 2:
        key, meaning = "TF", "expected true, predicted false - judged false too easily"
    elif tf or ft:
        return "errors both ways"
    else:
        return None

    flag = f"mostly {key} ({meaning})"
    counts = [run[rule][key] if rule in run else None for run in history]
    rising = _trailing_streak(counts, rising=True)
    if rising >= _MIN_STREAK:
        shown = counts[-(rising + 1) :]
        flag += f", {key} RISING {' -> '.join(str(c) for c in shown)}"
    return flag


def _trend_flags(history: list[RuleResults], rule: str, tolerance: float) -> str:
    series = _accuracy_series(history, rule)
    flags: list[str] = []

    falling = _trailing_streak(series, rising=False)
    if falling >= _MIN_STREAK:
        flags.append(f"FALLING {falling} runs in a row")

    best = _best_run(series)
    latest = series[-1]
    runs_since_best = len(series) - 1 - best
    if latest is not None and runs_since_best and latest < series[best] - tolerance:
        flags.append(f"below its best for {runs_since_best} run(s)")

    direction = _direction_flag(history, rule)
    if direction:
        flags.append(direction)

    return "; ".join(flags) or "-"


def format_rule_trend(history: list[RuleResults], tolerance: float) -> str:
    """`history` holds one RuleResults per run, oldest first. Returns "" when
    no run reported any per-rule accuracy (e.g. a single-criterion chain)."""
    rules = list(dict.fromkeys(rule for run in history for rule in run))
    if not rules:
        return ""

    run_headers = " | ".join(f"Run-{i}" for i in range(len(history)))
    lines = [
        "\n<-----------------RULE-TREND----------------------->",
        "Every checked rule across every run so far. Cell = accuracy, then TF/FT "
        "(TF = expected true, predicted false; FT = expected false, predicted true).",
        f"rule | {run_headers} | best | trend",
        "---|" + "---|" * len(history) + "---|---",
    ]
    for rule in rules:
        cells = " | ".join(_format_cell(run.get(rule)) for run in history)
        best = _best_run(_accuracy_series(history, rule))
        lines.append(f"{rule} | {cells} | Run-{best} | {_trend_flags(history, rule, tolerance)}")
    return "\n".join(lines) + "\n"
