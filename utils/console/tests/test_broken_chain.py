import importlib

import pytest

import chain_checker.utils.console.broken_chain as broken_chain

# console/__init__.py re-exports link_print (a function) under the same name
# as its own submodule, which shadows the submodule attribute on the
# `console` package - importlib gets the real module regardless.
_link_print_module = importlib.import_module("chain_checker.utils.console.link_print")


@pytest.fixture(autouse=True)
def _reset_glyph_cycle(monkeypatch):
    # The message is itself printed via link_print, whose glyph prefix
    # alternates between two patterns - one of which ("  O   O  ") contains
    # the art's own first line ("  O   O") as a substring. Pinning the cycle
    # keeps that prefix off the wide glyph, so a line search below can't
    # mistake the message's own prefix for the start of the art.
    monkeypatch.setattr(_link_print_module, "_LINK_STATE", 0)


def test_the_looks_broken_message_prints_before_the_art(capsys):
    broken_chain.print_broken_chain()

    out = capsys.readouterr().out
    art_lines = broken_chain._BROKEN_CHAIN_ART.split("\n")
    message_pos = out.index("Looks like the chain is broken")
    first_art_line = next(line for line in art_lines if line.strip())
    art_pos = out.index(first_art_line)

    assert message_pos < art_pos


def test_every_art_line_is_printed(capsys):
    broken_chain.print_broken_chain()

    out = capsys.readouterr().out
    for line in broken_chain._BROKEN_CHAIN_ART.split("\n"):
        assert line in out


def test_the_message_is_printed_exactly_once(capsys):
    broken_chain.print_broken_chain()

    out = capsys.readouterr().out
    assert out.count("Looks like the chain is broken") == 1
