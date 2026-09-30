import bisect
import shutil

from chain_checker.utils.console.colors import (
    DIM,
    LOGO_GRAY,
    LOGO_MID_GRAY,
    RESET,
    rgb,
    visible_width,
)

# --- 5x5 block font, just the letters CHAIN CHECKER needs -----------------

_FONT = {
    "A": [".###.", "#...#", "#####", "#...#", "#...#"],
    "C": [".####", "#....", "#....", "#....", ".####"],
    "E": ["#####", "#....", "###..", "#....", "#####"],
    "H": ["#...#", "#...#", "#####", "#...#", "#...#"],
    "I": ["#####", "..#..", "..#..", "..#..", "#####"],
    "K": ["#...#", "#..#.", "###..", "#..#.", "#...#"],
    "N": ["#...#", "##..#", "#.#.#", "#..##", "#...#"],
    "R": ["####.", "#...#", "####.", "#..#.", "#...#"],
    " ": ["...", "...", "...", "...", "..."],
}


def render_banner_text(text: str) -> list[str]:
    glyphs = [_FONT[ch] for ch in text.upper()]
    rows = ["" for _ in range(5)]
    color = rgb(LOGO_GRAY)
    for i, glyph in enumerate(glyphs):
        for row_index in range(5):
            block = glyph[row_index].replace("#", "█").replace(".", " ")
            rows[row_index] += f"{color}{block}{RESET} "
    return rows


# --- two chain-link rings, one wrapped around each word -------------------


def _ring_perimeter(width: int, height: int) -> list[tuple[int, int]]:
    top = [(0, c) for c in range(1, width - 1)]
    right = [(r, width - 1) for r in range(1, height - 1)]
    bottom = [(height - 1, c) for c in range(width - 2, 0, -1)]
    left = [(r, 0) for r in range(height - 2, 0, -1)]
    return top + right + bottom + left


def _reveal_order(
    perimeter: list[tuple[int, int]], start: tuple[int, int]
) -> list[tuple[int, int]]:
    i = perimeter.index(start)
    return perimeter[i:] + perimeter[:i]


_HORIZONTAL_STEP_COST = 1.0
_VERTICAL_STEP_COST = 3.2


def _cumulative_costs(order: list[tuple[int, int]], height: int) -> list[float]:
    cum = [0.0]
    for r, _ in order:
        cost = _HORIZONTAL_STEP_COST if r in (0, height - 1) else _VERTICAL_STEP_COST
        cum.append(cum[-1] + cost)
    return cum


def _reveal_count(order: list[tuple[int, int]], cum: list[float], t: float) -> int:
    return min(len(order), bisect.bisect_right(cum, t * cum[-1]) - 1)


def _render_word_ring(
    word_rows: list[str], revealed: set[tuple[int, int]], color: tuple[int, int, int]
) -> list[str]:
    width = visible_width(word_rows[0]) + 2
    height = len(word_rows) + 2
    code = rgb(color)
    rows = []
    for r in range(height):
        if r == 0 or r == height - 1:
            row_symbols = ""
            for c in range(width):
                if 1 <= c <= width - 2 and (r, c) in revealed:
                    if (c - 1) % 3 == 0:
                        s = "c"
                    elif (c - 1) % 3 == 1:
                        s = "ↄ"
                    else:
                        s = "-"
                    row_symbols += f"{code}{s}{RESET}"
                else:
                    row_symbols += " "

            rows.append(row_symbols)
            continue
        side_char = "0" if (r - 1) % 2 == 0 else "|"
        left = f"{code}{side_char}{RESET}" if (r, 0) in revealed else " "
        right = f"{code}{side_char}{RESET}" if (r, width - 1) in revealed else " "
        rows.append(left + word_rows[r - 1] + right)
    return rows


def build_logo_parts():
    chain_rows = render_banner_text("CHAIN")
    checker_rows = render_banner_text("CHECKER")
    chain_width = visible_width(chain_rows[0]) + 2
    checker_width = visible_width(checker_rows[0]) + 2
    ring_height = len(chain_rows) + 2

    chain_order = _reveal_order(_ring_perimeter(chain_width, ring_height), (ring_height // 2, 0))
    checker_order = _reveal_order(
        _ring_perimeter(checker_width, ring_height), (ring_height // 2, checker_width - 1)
    )
    chain_cum = _cumulative_costs(chain_order, ring_height)
    checker_cum = _cumulative_costs(checker_order, ring_height)
    return chain_rows, checker_rows, chain_order, checker_order, chain_cum, checker_cum, ring_height


_TAGLINE = "Harnessing the power of LLMs in the loop."


def logo_frame_lines(t: float, term_width: int, parts) -> list[str]:
    (
        chain_rows,
        checker_rows,
        chain_order,
        checker_order,
        chain_cum,
        checker_cum,
        ring_height,
    ) = parts
    left_ring = _render_word_ring(
        chain_rows, set(chain_order[: _reveal_count(chain_order, chain_cum, t)]), LOGO_MID_GRAY
    )
    right_ring = _render_word_ring(
        checker_rows,
        set(checker_order[: _reveal_count(checker_order, checker_cum, t)]),
        LOGO_MID_GRAY,
    )
    combined_lines = [left_ring[i] + right_ring[i] for i in range(ring_height)]
    pad = max(0, (term_width - visible_width(combined_lines[0])) // 2)
    combined = [" " * pad + line for line in combined_lines]

    seam_col = pad + visible_width(left_ring[0])
    gap_index = _TAGLINE.index(" of ")
    tagline_pad = max(0, seam_col - gap_index)
    tagline = " " * tagline_pad + f"{DIM}{_TAGLINE}{RESET}"

    return ["", *combined, tagline, ""]


def print_logo() -> None:
    term_width = shutil.get_terminal_size(fallback=(80, 24)).columns
    for line in logo_frame_lines(1.0, term_width, build_logo_parts()):
        print(line)
