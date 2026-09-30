import math
from typing import Any, Literal

from chain_checker.baseclasses.corpus.record import Record

_FLOAT_ABS_TOL = 1e-9


Kind = Literal["bool", "number", "bools", "value"]


class Label(Record):
    """The expected output values for a corpus case."""

    def set(self, data: dict[str, Any]) -> None:
        self._replace(data)

    def matches(self, other: Record) -> dict[str, bool]:
        other_data = other.get()
        # A key missing from `other` (e.g. a chain that didn't produce a
        # field the corpus expects) counts as a mismatch rather than raising,
        # so one absent key doesn't stop the rest of the comparison.
        return {
            key: key in other_data and values_agree(value, other_data[key])
            for key, value in self._data.items()
        }


def is_number(value: Any) -> bool:
    # bool is a subclass of int in Python, so isinstance(True, int) is True;
    # exclude it explicitly or a boolean field would be misclassified as numeric.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def value_kind(value: Any) -> Kind:
    if isinstance(value, bool):
        return "bool"
    if is_number(value):
        return "number"

    # A non-empty dict of all-bool values (e.g. named sub-rule verdicts) is
    # its own kind so it can be scored as one boolean per sub-rule, distinct
    # from a "value" dict that's compared as a single opaque unit.
    if isinstance(value, dict) and value and all(isinstance(v, bool) for v in value.values()):
        return "bools"
    return "value"


def describe_value(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 40 else f"{text[:37]}..."


def kind_conflict_message(
    key: str,
    first_id: str | int,
    first_value: Any,
    second_id: str | int,
    second_value: Any,
    path: str | None = None,
) -> str:

    where = f"Corpus file '{path}': output" if path else "Corpus output"
    return (
        f"{where} key '{key}' changes type between cases.\n"
        f"  Case '{first_id}': {describe_value(first_value)}\n"
        f"  Case '{second_id}': {describe_value(second_value)}\n"
        f"  Every case that checks a key has to agree on what kind of "
        f"value it holds, otherwise the two cannot be scored together - "
        f"an empty mapping counts as its own kind, since it is one whole "
        f"comparison rather than one sub-rule per key.\n"
        f"  Fix the corpus .yaml so '{key}' has one consistent type."
    )


def values_agree(expected: Any, actual: Any) -> bool:
    # Checked before the numeric branch below: bool is a subclass of int, so
    # without this a bool would fall through and 1 == True could register as
    # a numeric match against a value that was never meant to be boolean.
    if isinstance(expected, bool) or isinstance(actual, bool):
        return isinstance(expected, bool) and isinstance(actual, bool) and expected == actual

    if isinstance(expected, float) or isinstance(actual, float):
        if not isinstance(expected, (int, float)) or not isinstance(actual, (int, float)):
            return False
        return math.isclose(expected, actual, abs_tol=_FLOAT_ABS_TOL)

    if isinstance(expected, dict) or isinstance(actual, dict):
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            return False
        return expected.keys() == actual.keys() and all(
            values_agree(value, actual[key]) for key, value in expected.items()
        )

    if isinstance(expected, list) or isinstance(actual, list):
        if not isinstance(expected, list) or not isinstance(actual, list):
            return False
        return len(expected) == len(actual) and all(
            values_agree(want, got) for want, got in zip(expected, actual)
        )

    return expected == actual
