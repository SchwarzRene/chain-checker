import importlib

import chain_checker.utils.console.wait_bar as wait_bar
from chain_checker.utils.console.link_print import link_print
from chain_checker.utils.console.wait_bar import WaitBar

# console/__init__.py re-exports a function also named link_print, which
# shadows the submodule attribute on the `console` package - importlib gets
# the real module regardless (see test_link_print.py's own note on this).
link_print_module = importlib.import_module("chain_checker.utils.console.link_print")


def test_tick_prints_the_prefix_the_bar_and_the_elapsed_time(monkeypatch, capsys):
    monkeypatch.setattr(wait_bar, "IS_TTY", False)
    bar = WaitBar(width=5)

    bar.tick("my-prefix", 3.0)

    out = capsys.readouterr().out
    assert out.startswith("my-prefix [")
    assert "3s / record 3s" in out


def test_tick_switches_to_new_record_pace_once_it_beats_a_real_record(monkeypatch, capsys):
    monkeypatch.setattr(wait_bar, "IS_TTY", False)
    bar = WaitBar()
    bar.finish("p", 40.0)  # a real earlier finish - record is now 40, not just the default
    capsys.readouterr()

    bar.tick("p", 41.0)

    assert "new record pace" in capsys.readouterr().out


def test_tick_against_only_the_synthetic_default_stays_plain(monkeypatch, capsys):
    monkeypatch.setattr(wait_bar, "IS_TTY", False)
    bar = WaitBar()

    bar.tick("p", 31.0)  # past the default 30s scale, but nothing real to beat yet

    assert "new record pace" not in capsys.readouterr().out


def test_finish_defaults_to_a_finished_in_message_without_a_summary(monkeypatch, capsys):
    monkeypatch.setattr(wait_bar, "IS_TTY", False)
    bar = WaitBar()

    bar.finish("p", 5.0)

    assert "finished in 5s" in capsys.readouterr().out


def test_finish_uses_the_given_summary_instead_of_the_default(monkeypatch, capsys):
    monkeypatch.setattr(wait_bar, "IS_TTY", False)
    bar = WaitBar()

    bar.finish("p", 5.0, "done in 5s (avg 5s/entry, 0 left, ~0s remaining)")

    out = capsys.readouterr().out
    assert "done in 5s (avg 5s/entry, 0 left, ~0s remaining)" in out
    assert "finished in" not in out


def test_the_very_first_finish_is_never_flagged_a_new_record(monkeypatch, capsys):
    monkeypatch.setattr(wait_bar, "IS_TTY", False)
    bar = WaitBar()

    bar.finish("p", 45.0)  # well past the default 30s, but nothing real to beat yet

    assert "new record" not in capsys.readouterr().out


def test_a_later_finish_beating_a_real_record_is_flagged_even_without_ticking(monkeypatch, capsys):
    # A fast call can finish before the first tick ever fires - the new
    # record must still be caught by comparing directly against the record
    # as it stood going into this call.
    monkeypatch.setattr(wait_bar, "IS_TTY", False)
    bar = WaitBar()
    bar.finish("p", 40.0)
    capsys.readouterr()

    bar.finish("p", 45.0)

    assert "new record" in capsys.readouterr().out


def test_a_later_finish_short_of_the_record_is_not_flagged(monkeypatch, capsys):
    monkeypatch.setattr(wait_bar, "IS_TTY", False)
    bar = WaitBar()
    bar.finish("p", 40.0)
    capsys.readouterr()

    bar.finish("p", 35.0)

    assert "new record" not in capsys.readouterr().out


def test_histogram_marks_stack_as_a_count_rather_than_repeated_symbols(monkeypatch, capsys):
    monkeypatch.setattr(wait_bar, "IS_TTY", False)
    bar = WaitBar(width=5)
    bar.finish("p", 30.0)  # lands on the same (rightmost) column both times
    capsys.readouterr()

    bar.finish("p", 30.0)

    out = capsys.readouterr().out
    bar_text = out.split("[", 1)[1].split("]", 1)[0]
    assert bar_text == "····2"


def test_histogram_caps_at_a_plus_once_ten_marks_land_in_one_spot(monkeypatch, capsys):
    monkeypatch.setattr(wait_bar, "IS_TTY", False)
    bar = WaitBar(width=5)
    for _ in range(10):
        bar.finish("p", 30.0)

    last_line = capsys.readouterr().out.strip().splitlines()[-1]
    bar_text = last_line.split("[", 1)[1].split("]", 1)[0]
    assert bar_text == "····+"


def test_earlier_marks_recede_as_a_later_call_grows_the_record(monkeypatch, capsys):
    monkeypatch.setattr(wait_bar, "IS_TTY", False)
    bar = WaitBar(width=5)
    bar.finish("p", 20.0)  # first real mark becomes the record itself (fraction 1.0)
    first_bar_text = capsys.readouterr().out.split("[", 1)[1].split("]", 1)[0]
    assert first_bar_text == "····1"

    bar.finish("p", 40.0)  # a new, larger record — 20s is now only half of it

    second_bar_text = capsys.readouterr().out.split("[", 1)[1].split("]", 1)[0]
    # The 20s mark recedes from the last slot to the middle: re-projected
    # against the new 40s record, it shrinks toward the start rather than
    # staying put.
    assert second_bar_text == "··1·1"


def test_a_tty_run_redraws_in_place_and_only_finish_ends_the_line(monkeypatch, capsys):
    monkeypatch.setattr(wait_bar, "IS_TTY", True)
    bar = WaitBar()

    bar.tick("p", 1.0)
    tick_out = capsys.readouterr().out
    assert tick_out.startswith("\r\x1b[2K")
    assert not tick_out.endswith("\n")

    bar.finish("p", 2.0)
    finish_out = capsys.readouterr().out
    assert finish_out.startswith("\r\x1b[2K")
    assert finish_out.endswith("\n")


def test_a_tick_left_open_on_a_tty_forces_the_next_link_print_onto_a_fresh_line(
    monkeypatch, capsys
):
    # Regression: a real chain call fires its own link_print()s (e.g. the
    # analyser's "sent N message(s)..."/"reply received...") while the bar
    # is mid-wait - those must not land glued onto the bar's still-open line.
    monkeypatch.setattr(wait_bar, "IS_TTY", True)
    monkeypatch.setattr(link_print_module, "_LINK_STATE", 0)
    monkeypatch.setattr(link_print_module, "_WAIT_BAR_LINE_OPEN", False)
    bar = WaitBar()

    bar.tick("p", 1.0)
    link_print("(MODEL) reply received")

    out = capsys.readouterr().out
    assert not out.split("\n", 1)[0].endswith("(MODEL) reply received")
    assert out.rstrip("\n").endswith("(MODEL) reply received")


def test_a_tty_finish_leaves_no_line_open_for_the_next_print(monkeypatch, capsys):
    monkeypatch.setattr(wait_bar, "IS_TTY", True)
    monkeypatch.setattr(link_print_module, "_LINK_STATE", 0)
    monkeypatch.setattr(link_print_module, "_WAIT_BAR_LINE_OPEN", False)
    bar = WaitBar()

    bar.finish("p", 2.0)
    link_print("next")

    out = capsys.readouterr().out
    assert "\n\n" not in out
