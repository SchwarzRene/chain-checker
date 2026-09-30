import chain_checker.utils.console.colors as colors


def test_reset_and_dim_agree_with_the_current_color_support():
    # RESET/DIM are baked in at import time from SUPPORTS_COLOR, so their
    # values can't be asserted as literal strings without assuming a TTY -
    # only that they match the flag they were derived from.
    assert colors.RESET == ("\x1b[0m" if colors.SUPPORTS_COLOR else "")
    assert colors.DIM == ("\x1b[2m" if colors.SUPPORTS_COLOR else "")


def test_rgb_returns_nothing_when_color_is_not_supported(monkeypatch):
    monkeypatch.setattr(colors, "SUPPORTS_COLOR", False)

    assert colors.rgb((79, 109, 245)) == ""


def test_rgb_encodes_the_true_color_escape_when_supported(monkeypatch):
    monkeypatch.setattr(colors, "SUPPORTS_COLOR", True)

    assert colors.rgb((79, 109, 245)) == "\x1b[38;2;79;109;245m"


def test_visible_width_counts_plain_text():
    assert colors.visible_width("hello") == 5


def test_visible_width_ignores_embedded_ansi_codes(monkeypatch):
    # Force real escape codes regardless of the test runner's own TTY state,
    # so the assertion actually exercises the stripping regex rather than
    # comparing "hi" to itself.
    monkeypatch.setattr(colors, "SUPPORTS_COLOR", True)
    text = f"{colors.rgb((1, 2, 3))}hi\x1b[0m"

    assert colors.visible_width(text) == 2


def test_visible_width_of_an_empty_string_is_zero():
    assert colors.visible_width("") == 0
