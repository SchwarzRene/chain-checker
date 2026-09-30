import pytest

from chain_checker.modifier.model.compact_format.scalars import (
    format_inline_list,
    format_scalar,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, "true"),
        (False, "false"),
        (None, "null"),
        (3, "3"),
        ("hello", "hello"),
        (1.0, "1"),
        (0.5, "0.5"),
        (1.23456, "1.2346"),
        (1.2000, "1.2"),
    ],
    ids=[
        "true",
        "false",
        "none",
        "int",
        "str",
        "integral-float-drops-decimal",
        "simple-float",
        "rounds-to-four-places",
        "drops-trailing-zeros",
    ],
)
def test_format_scalar_renders_each_kind_of_value(value, expected):
    assert format_scalar(value) == expected


def test_a_bool_is_never_read_as_a_float():
    # bool is a subclass of int/float-comparable in Python; the bool check
    # must win so True never prints as "1".
    assert format_scalar(True) != format_scalar(1.0)


@pytest.mark.parametrize(
    "value, expected",
    [(float("nan"), "nan"), (float("inf"), "inf"), (float("-inf"), "-inf")],
    ids=["nan", "inf", "-inf"],
)
def test_a_non_finite_float_renders_instead_of_crashing(value, expected):
    # int(round(value, 4)) raises ValueError on nan and OverflowError on
    # +/-inf - a degenerate metric must still produce a report line.
    assert format_scalar(value) == expected


def test_format_inline_list_of_an_empty_list_is_explicit_not_blank():
    assert format_inline_list([]) == "[]"


def test_format_inline_list_renders_each_value():
    assert format_inline_list([1, 2.5, None]) == "[1, 2.5, null]"


def test_format_inline_list_collapses_a_constant_run():
    assert format_inline_list([True, True, True]) == "[true] x3 (constant)"


def test_format_inline_list_does_not_collapse_a_single_value():
    # len(values) > 1 guards this: one repeated value isn't "a run" of
    # anything yet, so it should print like any other one-item list.
    assert format_inline_list([True]) == "[true]"


def test_format_inline_list_does_not_collapse_values_that_merely_differ():
    assert format_inline_list([1, 1, 2]) == "[1, 1, 2]"
