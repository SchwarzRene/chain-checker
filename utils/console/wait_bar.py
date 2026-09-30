import sys

from chain_checker.baseclasses.loop.progress import format_duration
from chain_checker.utils.console.colors import IS_TTY, RESET, SUPPORTS_COLOR
from chain_checker.utils.console.link_print import (
    note_wait_bar_line_closed,
    note_wait_bar_line_open,
)

# Placeholder scale before any call finishes; not a real record yet.
_DEFAULT_RECORD_SECONDS = 1.0

# Bold, so the moving marker reads as distinct from the histogram digits.
_LIVE_COLOR = "\x1b[1m" if SUPPORTS_COLOR else ""


class WaitBar:
    """A single line, redrawn in place, that races each call against the
    longest call seen so far (the "record"), turning finished calls into a
    running histogram.

    Marks are stored as raw seconds but rendered relative to the current
    record (`_column`), so they recede left whenever a new record stretches
    the scale. `tick`/`finish` take the caller's own `prefix` text so the bar
    and its label print as one line.
    """

    def __init__(self, width: int = 40) -> None:
        self._width = width
        self._record = _DEFAULT_RECORD_SECONDS
        self._marks: list[float] = []
        self._live_new_record = False

    def _column(self, seconds: float) -> int:
        fraction = min(1.0, seconds / self._record) if self._record > 0 else 1.0
        return round(fraction * (self._width - 1))

    def _histogram_cells(self) -> list[str]:
        counts = [0] * self._width
        for mark in self._marks:
            counts[self._column(mark)] += 1

        cells = []
        for count in counts:
            if count == 0:
                cells.append("·")
            elif count < 10:
                cells.append(str(count))
            else:
                cells.append("+")
        return cells

    def _render(self, prefix: str, suffix: str, elapsed: float | None) -> str:
        cells = self._histogram_cells()
        if elapsed is not None:
            cells[self._column(elapsed)] = f"{_LIVE_COLOR}x{RESET}"
        bar = "".join(cells)
        return f"{prefix} [{bar}] {suffix}"

    def _write(self, line: str, *, final: bool) -> None:
        if IS_TTY:
            sys.stdout.write("\r\x1b[2K" + line)
            if final:
                sys.stdout.write("\n")
                note_wait_bar_line_closed()
            else:
                # No trailing newline: the next redraw overwrites this line
                # instead of starting a new one.
                note_wait_bar_line_open()
        else:
            sys.stdout.write(line + "\n")
            note_wait_bar_line_closed()
        sys.stdout.flush()

    def tick(self, prefix: str, elapsed: float) -> None:
        if elapsed > self._record:
            self._record = elapsed
            # Only a real record if an earlier call already finished.
            if self._marks:
                self._live_new_record = True

        if self._live_new_record:
            suffix = f"{format_duration(elapsed)} - new record pace"
        else:
            suffix = f"{format_duration(elapsed)} / record {format_duration(self._record)}"
        self._write(self._render(prefix, suffix, elapsed), final=False)

    def finish(self, prefix: str, elapsed: float, summary: str = "") -> None:
        # `_live_new_record` only catches calls that ticked; a call that
        # finishes without ever ticking still needs its own record check.
        is_new_record = self._live_new_record or (bool(self._marks) and elapsed > self._record)
        self._marks.append(elapsed)
        self._record = max(self._record, elapsed)
        self._live_new_record = False

        suffix = summary if summary else f"finished in {format_duration(elapsed)}"
        if is_new_record:
            suffix += " - new record"
        self._write(self._render(prefix, suffix, None), final=True)
