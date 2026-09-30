import sys

from chain_checker.utils.console.colors import LOGO_MID_GRAY, RESET, rgb

# Cycled by _next_glyph() across successive calls, so a run of link_print()
# lines simulates one continuous, wider chain rather than repeating the same
# link shape on every line.
_LINK_STATE = 0


_WAIT_BAR_LINE_OPEN = False


def note_wait_bar_line_open() -> None:
    global _WAIT_BAR_LINE_OPEN
    _WAIT_BAR_LINE_OPEN = True


def note_wait_bar_line_closed() -> None:
    global _WAIT_BAR_LINE_OPEN
    _WAIT_BAR_LINE_OPEN = False


def _close_any_open_wait_bar_line() -> None:
    global _WAIT_BAR_LINE_OPEN
    if _WAIT_BAR_LINE_OPEN:
        sys.stdout.write("\n")
        _WAIT_BAR_LINE_OPEN = False


def _next_glyph() -> str:
    global _LINK_STATE

    glyph = "    0    " if _LINK_STATE % 5 == 0 else "  O   O  "
    _LINK_STATE += 1
    return glyph


def next_glyph() -> str:
    """The same cycling, pre-colored glyph `link_print` prefixes its own lines
    with, for a caller (like `WaitBar`) that redraws its own line but still
    wants it to look like part of the same chain."""
    return f"{rgb(LOGO_MID_GRAY)}{_next_glyph()}{RESET}"


def link_print(*values, sep: str = " ", end: str = "\n") -> None:
    """Drop-in replacement for print() that prefixes all lines - including
    empty ones - with a vertical chain pattern."""
    _close_any_open_wait_bar_line()
    text = sep.join(str(v) for v in values) if values else ""
    lines = text.split("\n")

    formatted_lines = []
    for line in lines:
        prefix = f"{rgb(LOGO_MID_GRAY)}{_next_glyph()}{RESET}"
        suffix = f" {line}" if line else ""
        formatted_lines.append(f"{prefix}{suffix}")

    sys.stdout.write("\n".join(formatted_lines) + end)
    sys.stdout.flush()


def link_print_warning(message: str) -> None:
    """Prints a `(TAG) WARNING: ...`-style message consistently everywhere -
    a blank line above it, and a `---` mark run directly against its tag so
    it stands out while scrolling past normal output."""
    link_print()
    link_print(f"---{message}")
