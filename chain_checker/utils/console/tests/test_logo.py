import os

import pytest

import chain_checker.utils.console.logo as logo
from chain_checker.utils.console.colors import visible_width


def test_render_banner_text_returns_one_row_per_font_line():
    assert len(logo.render_banner_text("A")) == 5


def test_the_font_is_case_insensitive():
    assert logo.render_banner_text("chain") == logo.render_banner_text("CHAIN")


def test_each_letter_adds_the_same_fixed_width():
    one_letter = visible_width(logo.render_banner_text("A")[0])
    two_letters = visible_width(logo.render_banner_text("AA")[0])

    assert two_letters == one_letter * 2


def test_a_character_outside_the_font_is_rejected():
    with pytest.raises(KeyError):
        logo.render_banner_text("B")


def test_build_logo_parts_returns_a_cost_entry_per_perimeter_cell_plus_one():
    parts = logo.build_logo_parts()
    chain_order, chain_cum = parts[2], parts[4]

    # _cumulative_costs starts the running total at 0.0 before the first
    # cell, so it always has one more entry than the order it costs.
    assert len(chain_cum) == len(chain_order) + 1


def test_ring_height_is_the_font_height_plus_a_border_on_each_side():
    parts = logo.build_logo_parts()

    assert parts[-1] == 5 + 2


def test_at_t_zero_the_top_ring_border_is_not_drawn_yet():
    # Only the ring's own border cells (top/bottom rows, and the corner
    # chars of the middle rows) are gated by reveal progress - the banner
    # letters those middle rows carry are drawn unconditionally. The top
    # border row is pure border, so it is the one row guaranteed blank
    # before anything has been revealed.
    parts = logo.build_logo_parts()

    top_border_row = logo.logo_frame_lines(0.0, 100, parts)[1]

    assert top_border_row.strip() == ""


def test_at_t_one_the_top_ring_border_is_fully_drawn():
    parts = logo.build_logo_parts()

    top_border_row = logo.logo_frame_lines(1.0, 100, parts)[1]

    assert top_border_row.strip() != ""


def test_the_frame_width_is_unchanged_by_how_much_of_the_ring_is_revealed():
    parts = logo.build_logo_parts()

    empty_frame = logo.logo_frame_lines(0.0, 100, parts)
    full_frame = logo.logo_frame_lines(1.0, 100, parts)

    # Revealing a cell swaps a space for a symbol of the same width - the
    # ring's footprint on screen never changes size, only its content.
    assert [visible_width(line) for line in empty_frame] == [
        visible_width(line) for line in full_frame
    ]


def test_the_frame_has_a_leading_and_trailing_blank_line():
    parts = logo.build_logo_parts()

    frame = logo.logo_frame_lines(1.0, 100, parts)

    assert frame[0] == ""
    assert frame[-1] == ""


def test_the_frame_carries_one_line_per_ring_row_plus_the_tagline():
    parts = logo.build_logo_parts()
    ring_height = parts[-1]

    frame = logo.logo_frame_lines(1.0, 100, parts)

    assert len(frame) == ring_height + 3


def test_the_tagline_survives_inside_the_second_to_last_line():
    parts = logo.build_logo_parts()

    frame = logo.logo_frame_lines(1.0, 100, parts)

    assert logo._TAGLINE in frame[-2]


def test_print_logo_writes_the_fully_revealed_frame(monkeypatch, capsys):
    monkeypatch.setattr(
        logo.shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((100, 24))
    )

    logo.print_logo()

    expected = logo.logo_frame_lines(1.0, 100, logo.build_logo_parts())
    assert capsys.readouterr().out.splitlines() == expected
