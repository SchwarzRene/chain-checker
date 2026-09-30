import pytest

from chain_checker.utils.report.formatting import (
    format_metric_value,
    format_value,
    nice_step,
)

# ---- format_value ----


def test_format_value_renders_none_as_a_dash():
    assert format_value(None) == "-"


def test_format_value_renders_bools_as_lowercase_words_not_numbers():
    # bool is a subclass of int, so this must be checked before the general
    # numeric case - otherwise True/False would render as "1"/"0".
    assert format_value(True) == "true"
    assert format_value(False) == "false"


def test_format_value_renders_floats_without_trailing_zeros():
    assert format_value(0.800) == "0.8"


def test_format_value_falls_back_to_str_for_everything_else():
    assert format_value(7) == "7"
    assert format_value("formal") == "formal"


# ---- format_metric_value ----


@pytest.mark.parametrize(
    "value, expected",
    [
        (150, "150"),
        (1234, "1,234"),
        (7.5, "7.50"),
        (1.0, "1.00"),
        (0.3333, "0.333"),
        (0.0, "0.000"),
    ],
)
def test_format_metric_value_picks_precision_by_magnitude(value, expected):
    assert format_metric_value(value) == expected


def test_format_metric_value_handles_negative_numbers_by_magnitude():
    assert format_metric_value(-150) == "-150"
    assert format_metric_value(-0.5) == "-0.500"


# ---- nice_step ----


def test_nice_step_rounds_up_to_the_nearest_1_2_5_step():
    assert nice_step(80, target_lines=8) == 10
    assert nice_step(40, target_lines=8) == 5
    assert nice_step(16, target_lines=8) == 2


def test_nice_step_rounds_a_non_round_spacing_up_not_down():
    # raw_step = 137/8 = 17.125, which must round up to 20, not down to 10.
    assert nice_step(137, target_lines=8) == 20


def test_nice_step_is_stable_across_orders_of_magnitude():
    assert nice_step(8000, target_lines=8) == 1000


def test_nice_step_returns_one_for_a_non_positive_max():
    assert nice_step(0) == 1.0
    assert nice_step(-5) == 1.0
