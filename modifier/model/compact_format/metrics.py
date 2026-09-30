from typing import Any

from chain_checker.baseclasses.metrics.m_single_value import iter_rule_results
from chain_checker.modifier.model.compact_format.scalars import (
    format_inline_list,
    format_scalar,
)


def _is_confusion_dict(value: object) -> bool:
    return isinstance(value, dict) and {"TT", "TF", "FT", "FF"} <= value.keys()


def _is_numeric_metric_dict(value: object) -> bool:
    return isinstance(value, dict) and "mae" in value


def _format_confusion_table(rows: list[tuple[str, dict[str, Any]]]) -> str:
    lines = ["field | accuracy | TT | TF | FT | FF", "---|---|---|---|---|---"]
    for key, value in rows:
        lines.append(
            f"{key} | {format_scalar(value.get('accuracy'))} | "
            f"{value.get('TT')} | {value.get('TF')} | {value.get('FT')} | {value.get('FF')}"
        )
    return "\n".join(lines)


def _format_numeric_metric_line(key: str, value: dict[str, Any]) -> str:
    parts = [f"mae={format_scalar(value.get('mae'))}"]
    for subkey, subvalue in value.items():
        if subkey == "mae":
            continue
        parts.append(
            f"{subkey}={format_inline_list(subvalue)}"
            if isinstance(subvalue, list)
            else f"{subkey}={format_scalar(subvalue)}"
        )
    return f"{key}: " + "  ".join(parts)


def _append_rule_rows(
    results: dict[str, Any],
    table_rows: list[tuple[str, dict[str, Any]]],
    other_lines: list[str],
) -> None:
    for path, rule_results in iter_rule_results(results):
        key = ".".join(path)
        if _is_confusion_dict(rule_results):
            table_rows.append((key, rule_results))
        elif _is_numeric_metric_dict(rule_results):
            other_lines.append(_format_numeric_metric_line(key, rule_results))
        else:
            other_lines.append(f"{key}: {rule_results!r}")


def format_metric_block(name: str | None, results: Any) -> str:
    lines = [f"## {name}"]

    if isinstance(results, dict):
        table_rows, other_lines = [], []
        for key, value in results.items():
            if key == "rules" and isinstance(value, dict):
                # Accuracy-Metrics nests its per-rule results ({rule: {"results"}}
                # or {rule: {"sub_rules": ...}}) - flatten them into table rows
                # instead of printing the raw dict.
                _append_rule_rows(results, table_rows, other_lines)
            elif _is_confusion_dict(value):
                table_rows.append((key, value))
            elif _is_numeric_metric_dict(value):
                other_lines.append(_format_numeric_metric_line(key, value))
            elif isinstance(value, dict):
                other_lines.append(f"{key}: {value!r}")
            elif isinstance(value, list):
                other_lines.append(f"{key}: {format_inline_list(value)}")
            else:
                other_lines.append(f"{key}: {format_scalar(value)}")
        if table_rows:
            lines.append(_format_confusion_table(table_rows))
        lines.extend(other_lines)
    elif isinstance(results, list):
        lines.append(format_inline_list(results))
    else:
        lines.append(format_scalar(results))

    return "\n".join(lines)


def format_accuracy_report(items: list[dict[str, Any]]) -> str:
    return "\n\n".join(format_metric_block(item.get("name"), item.get("results")) for item in items)
