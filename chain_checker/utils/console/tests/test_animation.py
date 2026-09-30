import os

import pytest

import chain_checker.utils.console.animation as animation
from chain_checker.utils.console.logo import build_logo_parts, logo_frame_lines


@pytest.fixture(autouse=True)
def _no_real_delays(monkeypatch):
    # animate_chain paces itself with real time.sleep() calls (up to ~1.5s
    # total across its two phases); tests care about what gets drawn, not how
    # long it takes to draw it.
    monkeypatch.setattr(animation.time, "sleep", lambda seconds: None)


@pytest.fixture(autouse=True)
def _fixed_terminal_width(monkeypatch):
    monkeypatch.setattr(
        animation.shutil,
        "get_terminal_size",
        lambda fallback=(80, 24): os.terminal_size((100, 24)),
    )


def test_chain_segment_cycles_through_the_three_link_characters():
    assert animation._chain_segment(0, 6) == "cↄ-cↄ-"


def test_chain_segment_offsets_its_cycle_by_the_starting_column():
    assert animation._chain_segment(1, 3) == "ↄ-c"


def test_overlay_chain_is_spliced_into_leading_whitespace_only():
    line = "    XYZ"

    result = animation._overlay_chain(line, 1, "c", "")

    assert result == f" c{animation.RESET}  XYZ"
    assert result.endswith("XYZ")


def test_overlay_chain_onto_an_all_blank_line_just_places_the_glyph():
    assert animation._overlay_chain("      ", 2, "cↄ", "") == f"  cↄ{animation.RESET}"


def test_overlay_chain_truncates_a_glyph_that_would_overrun_the_margin():
    line = "  XYZ"

    result = animation._overlay_chain(line, 0, "cↄ-cↄ", "")

    assert result == f"cↄ{animation.RESET}XYZ"


def test_a_non_tty_run_prints_the_fully_revealed_frame_exactly_once(monkeypatch, capsys):
    monkeypatch.setattr(animation, "IS_TTY", False)

    animation.animate_chain()

    expected = logo_frame_lines(1.0, 100, build_logo_parts())
    assert capsys.readouterr().out.splitlines() == expected


def test_a_non_tty_run_pauses_once_after_printing(monkeypatch):
    monkeypatch.setattr(animation, "IS_TTY", False)
    calls = []
    monkeypatch.setattr(animation.time, "sleep", lambda seconds: calls.append(seconds))

    animation.animate_chain()

    assert calls == [animation._POST_ANIMATION_PAUSE]


def test_a_tty_run_completes_without_raising_and_draws_something(monkeypatch, capsys):
    monkeypatch.setattr(animation, "IS_TTY", True)

    animation.animate_chain(duration=0.01)

    assert capsys.readouterr().out != ""


def test_a_tty_run_ends_with_the_tagline_still_on_screen(monkeypatch, capsys):
    # The chain-drop transition overlays glyphs onto a copy of the final,
    # fully-revealed frame - it never rewrites the logo itself, only the
    # leading whitespace in front of it.
    monkeypatch.setattr(animation, "IS_TTY", True)

    animation.animate_chain(duration=0.01)

    out = capsys.readouterr().out
    assert "Harnessing the power of LLMs in the loop." in out
