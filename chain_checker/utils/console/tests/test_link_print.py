import importlib

import pytest

from chain_checker.utils.console import colors
from chain_checker.utils.console.link_print import link_print, next_glyph

# console/__init__.py re-exports a function also named link_print, which
# shadows the submodule attribute on the `console` package - importlib gets
# the real module (keyed by its full dotted path in sys.modules) regardless
# of that shadowing, where `import ... as` or `from ... import` would not.
link_print_module = importlib.import_module("chain_checker.utils.console.link_print")


@pytest.fixture(autouse=True)
def _reset_glyph_cycle(monkeypatch):
    # _LINK_STATE is process-wide and increments on every call, so without a
    # reset each test would see whatever cycle position the previous test
    # left it at.
    monkeypatch.setattr(link_print_module, "_LINK_STATE", 0)


@pytest.fixture(autouse=True)
def _reset_wait_bar_line_state(monkeypatch):
    monkeypatch.setattr(link_print_module, "_WAIT_BAR_LINE_OPEN", False)


def _prefix(glyph: str) -> str:
    return f"{colors.rgb(colors.LOGO_MID_GRAY)}{glyph}{colors.RESET}"


def test_a_single_value_is_prefixed_and_ends_with_a_newline(capsys):
    link_print("hello")

    assert capsys.readouterr().out == f"{_prefix('    0    ')} hello\n"


def test_several_positional_values_are_joined_with_the_separator(capsys):
    link_print("a", "b", "c")

    assert capsys.readouterr().out == f"{_prefix('    0    ')} a b c\n"


def test_a_custom_separator_is_honoured(capsys):
    link_print("a", "b", sep="-")

    assert capsys.readouterr().out == f"{_prefix('    0    ')} a-b\n"


def test_a_custom_end_replaces_the_trailing_newline(capsys):
    link_print("hello", end="!")

    assert capsys.readouterr().out == f"{_prefix('    0    ')} hello!"


def test_multiline_text_gets_its_own_prefix_on_every_line(capsys):
    link_print("first\nsecond")

    lines = capsys.readouterr().out.rstrip("\n").split("\n")
    assert lines == [
        f"{_prefix('    0    ')} first",
        f"{_prefix('  O   O  ')} second",
    ]


def test_an_empty_line_still_gets_a_bare_prefix_with_no_trailing_space(capsys):
    link_print("first\n\nthird")

    lines = capsys.readouterr().out.rstrip("\n").split("\n")
    assert lines[1] == _prefix("  O   O  ")


def test_calling_with_no_arguments_prints_one_bare_prefixed_line(capsys):
    link_print()

    assert capsys.readouterr().out == f"{_prefix('    0    ')}\n"


def test_the_wide_glyph_recurs_every_fifth_line(capsys):
    for _ in range(6):
        link_print("x")

    lines = capsys.readouterr().out.rstrip("\n").split("\n")
    glyphs = [line.split(" x")[0] for line in lines]
    assert glyphs[0] == _prefix("    0    ")
    assert glyphs[1:5] == [_prefix("  O   O  ")] * 4
    assert glyphs[5] == _prefix("    0    ")


def test_next_glyph_matches_what_link_print_would_have_used():
    assert next_glyph() == _prefix("    0    ")


def test_next_glyph_shares_link_prints_own_cycle(capsys):
    next_glyph()  # consumes the first "0" glyph

    link_print("x")

    assert capsys.readouterr().out == f"{_prefix('  O   O  ')} x\n"


def test_a_line_left_open_by_a_wait_bar_is_closed_before_the_next_print(capsys):
    link_print_module.note_wait_bar_line_open()

    link_print("after")

    assert capsys.readouterr().out == f"\n{_prefix('    0    ')} after\n"


def test_nothing_extra_is_inserted_when_no_line_was_left_open(capsys):
    link_print("plain")

    assert capsys.readouterr().out == f"{_prefix('    0    ')} plain\n"


def test_closing_an_open_line_clears_the_flag_for_the_call_after(capsys):
    # The fix-up newline must only ever precede the *first* print after the
    # bar left a line open - a later, unrelated print must not get one too.
    link_print_module.note_wait_bar_line_open()
    link_print("first")

    link_print("second")

    out = capsys.readouterr().out
    assert out == f"\n{_prefix('    0    ')} first\n{_prefix('  O   O  ')} second\n"
