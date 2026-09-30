import math


def format_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        if not math.isfinite(value):
            # int(rounded) below raises ValueError on nan and OverflowError
            # on +/-inf - a degenerate metric (e.g. mae over zero cases)
            # must still render, not crash the whole report.
            return str(value)
        # Rounds to a human-scannable width and drops a trailing ".0000" (or
        # trailing zeros), so an integral float reads as a plain integer.
        rounded = round(value, 4)
        if rounded == int(rounded):
            return str(int(rounded))
        return f"{rounded:.4f}".rstrip("0").rstrip(".")
    return str(value)


def format_inline_list(values: list[object]) -> str:
    if not values:
        return "[]"
    if len(values) > 1 and all(v == values[0] for v in values):
        # A metric repeating the same value across every case reads as noise
        # at any length; collapse it to the value once plus a count.
        return f"[{format_scalar(values[0])}] x{len(values)} (constant)"
    return "[" + ", ".join(format_scalar(v) for v in values) + "]"
