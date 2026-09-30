import shutil
import sys
import time
from typing import NamedTuple

from chain_checker.utils.console.colors import IS_TTY, LOGO_MID_GRAY, RESET, rgb
from chain_checker.utils.console.logo import build_logo_parts, logo_frame_lines

_POST_ANIMATION_PAUSE = 0.5

# Column the outgoing chains reach toward - matches the "0" in link_print's
# own vertical glyph ("    0    "), so the transition hands off seamlessly
# into the real output that follows.
_LEFT_MARGIN_COL = 4


def _chain_segment(start_col: int, length: int) -> str:
    out = ""
    for i in range(length):
        if (start_col + i) % 3 == 0:
            out += "c"
        elif (start_col + i) % 3 == 1:
            out += "ↄ"
        else:
            out += "-"

    return out


def _staircase_rows(
    leading: int, reach: int, start_row: int, num_rows: int
) -> list[tuple[int, int, str]]:
    """num_rows segments descending one row at a time from start_row, each
    covering an equal share of reach, landing on the left margin by the last
    row."""
    per_row = reach / num_rows
    rows = []
    for r in range(num_rows):
        seg_end = round(leading - r * per_row)
        seg_start = max(_LEFT_MARGIN_COL, round(leading - (r + 1) * per_row))
        length = max(1, seg_end - seg_start)
        rows.append((start_row + r, seg_start, _chain_segment(seg_start, length)))
    return rows


def _overlay_chain(line: str, col: int, glyph: str, color_code: str) -> str:
    """Splice a chain glyph into a frame line's leading whitespace, without
    disturbing whatever real content (ring border, letters, text) follows."""
    leading = len(line) - len(line.lstrip(" "))
    rest = line.lstrip(" ")
    if not rest:
        return (" " * col) + f"{color_code}{glyph}{RESET}"
    if col + len(glyph) > leading:
        glyph = glyph[: max(0, leading - col)]
    gap = max(0, leading - col - len(glyph))
    return (" " * col) + f"{color_code}{glyph}{RESET}" + (" " * gap) + rest


class _ChainPlan(NamedTuple):
    """A chain that stays flat on its own row for `flat_fraction` of its
    distance to the left margin, then steps down one row per segment with
    whatever's left, until it reaches the row before the caller's own
    output starts."""

    row: int
    leading: int
    flat_len: int
    flat_col: int
    flat_fraction: float
    staircase: list[tuple[int, int, str]]


def _plan_chain(row: int, leading: int, flat_fraction: float, first_line_row: int) -> _ChainPlan:
    reach = max(0, leading - _LEFT_MARGIN_COL)
    flat_len = round(reach * flat_fraction)
    flat_col = max(_LEFT_MARGIN_COL, leading - flat_len)
    descent_row = row + (1 if flat_len else 0)
    descent_rows = max(1, first_line_row - descent_row)
    staircase = _staircase_rows(flat_col, flat_col - _LEFT_MARGIN_COL, descent_row, descent_rows)
    return _ChainPlan(row, leading, flat_len, flat_col, flat_fraction, staircase)


def _draw_chain_step(
    frame: list[str], plan: _ChainPlan, step: int, drop_steps: int, color: str
) -> None:
    flat_phase_steps = max(1, round(drop_steps * plan.flat_fraction)) if plan.flat_len else 0

    if plan.flat_len:
        grown = min(plan.flat_len, round(plan.flat_len * step / flat_phase_steps))
        start_col = max(plan.flat_col, plan.leading - grown)
        glyph = _chain_segment(start_col, plan.leading - start_col)
        frame[plan.row] = _overlay_chain(frame[plan.row], start_col, glyph, color)

    if step > flat_phase_steps:
        revealed = round(
            len(plan.staircase) * (step - flat_phase_steps) / max(1, drop_steps - flat_phase_steps)
        )
        for row_idx, col, glyph in plan.staircase[:revealed]:
            frame[row_idx] = _overlay_chain(frame[row_idx], col, glyph, color)


def animate_chain(duration: float = 1.0) -> None:
    term_width = shutil.get_terminal_size(fallback=(80, 24)).columns
    parts = build_logo_parts()

    if not IS_TTY:
        for line in logo_frame_lines(1.0, term_width, parts):
            print(line)
        time.sleep(_POST_ANIMATION_PAUSE)
        return

    _, _, chain_order, checker_order, _, _, ring_height = parts
    block_height = len(logo_frame_lines(0.0, term_width, parts))
    steps = min(max(len(chain_order), len(checker_order)), 30)
    delay = duration / (steps + 1)

    redraw = False
    for i in range(steps + 1):
        if redraw:
            sys.stdout.write(f"\x1b[{block_height}F\x1b[0J")
        sys.stdout.write("\n".join(logo_frame_lines(i / steps, term_width, parts)) + "\n")
        sys.stdout.flush()
        redraw = True
        time.sleep(delay)

    # --- Three chains grow out of the ring at once, each running flat on its
    # own row for a fraction of its distance to the left margin before
    # stepping down toward the caller's first output line, splitting what's
    # left evenly across the rows it still has to cross. Top runs flat
    # longest (2/3), then middle (1/2), then bottom dives immediately (0). ---
    final_logo_lines = logo_frame_lines(1.0, term_width, parts)
    color_code = rgb(LOGO_MID_GRAY)

    top_row, middle_row, bottom_row = 1, 1 + ring_height // 2, ring_height
    row_leading = {
        row_idx: len(final_logo_lines[row_idx]) - len(final_logo_lines[row_idx].lstrip(" "))
        for row_idx in (top_row, middle_row, bottom_row)
    }
    first_line_row = len(final_logo_lines)  # where the caller's own link_print output starts

    # Drawn in this order (ascending flat_fraction): on any row their
    # staircases share, the chain that started descending earliest always
    # sits at a larger column, and _overlay_chain needs that content in
    # place as "rest" before a later chain carves in front of it.
    plans = [
        _plan_chain(bottom_row, row_leading[bottom_row], 0.0, first_line_row),
        _plan_chain(middle_row, row_leading[middle_row], 1 / 2, first_line_row),
        _plan_chain(top_row, row_leading[top_row], 2 / 3, first_line_row),
    ]

    total_rows = max(row_idx for plan in plans for row_idx, _, _ in plan.staircase) + 1
    top_reach = max(0, row_leading[top_row] - _LEFT_MARGIN_COL)
    drop_steps = max(1, min(top_reach, 12), *(len(plan.staircase) for plan in plans))
    drop_delay = _POST_ANIMATION_PAUSE / drop_steps
    current_block_height = block_height

    for step in range(1, drop_steps + 1):
        frame = final_logo_lines[:]

        while len(frame) < total_rows:
            frame.append("")

        for plan in plans:
            _draw_chain_step(frame, plan, step, drop_steps, color_code)

        if redraw:
            sys.stdout.write(f"\x1b[{current_block_height}F\x1b[0J")

        sys.stdout.write("\n".join(frame) + "\n")
        sys.stdout.flush()

        current_block_height = len(frame)
        redraw = True
        time.sleep(drop_delay)
