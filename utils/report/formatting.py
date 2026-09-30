import math
from typing import Any


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def format_metric_value(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def nice_step(max_value: float, target_lines: int = 8) -> float:
    """Round up to the nearest "nice" 1/2/5 x 10^n step, so a chart's y-axis
    gridlines land on round numbers instead of an arbitrary raw spacing."""
    if max_value <= 0:
        return 1.0

    raw_step = max_value / target_lines
    magnitude = 10 ** math.floor(math.log10(raw_step))
    for multiple in (1, 2, 5, 10):
        step = multiple * magnitude
        if step >= raw_step:
            return step
    return 10 * magnitude
