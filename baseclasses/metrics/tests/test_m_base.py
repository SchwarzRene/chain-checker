"""Unit tests for chain_checker/baseclasses/metrics/m_base.py.

Every metric, whatever it measures, is read back through the same three
keys - to_dict()'s name/description/results - since that is the shape both
ModifierMetrics and the epoch report expect. And since a subclass only ever
overrides the collector it actually uses, the other two have to keep
refusing rather than quietly no-op, or a metric fed the wrong way would
report a plausible empty result instead of surfacing the mistake.
"""

import typing
from typing import Any

import pytest

from chain_checker.baseclasses.corpus.entry import Entry
from chain_checker.baseclasses.metrics.m_base import BaseMetrics
from chain_checker.baseclasses.metrics.m_tokens import TokenUsageMetrics


class _Stub(BaseMetrics):
    """Minimal concrete metric - just enough of a compute() to be usable."""

    def __init__(self, name: str = "A-Metric", description: str = "what it measures") -> None:
        super().__init__(name, description)

    def compute(self) -> dict[str, Any]:
        return {"value": 1}


def test_name_and_description_are_reported_back():
    metric = _Stub()

    assert metric.get_name() == "A-Metric"
    assert metric.get_description() == "what it measures"


def test_to_dict_has_exactly_the_three_reported_keys():
    # Both consumers key off these exact names - the modifier looks up a
    # metric by "name", the report renders "description" beside "results" -
    # so any one of the three being renamed or dropped breaks a reader.
    assert set(_Stub().to_dict()) == {"name", "description", "results"}


def test_to_dict_calls_compute_rather_than_reading_a_stored_value():
    # A subclass's only override is compute(), so to_dict() has to call
    # through to it live each time rather than caching an earlier result.
    class Counting(BaseMetrics):
        def __init__(self) -> None:
            super().__init__("Counting", "counts compute() calls")
            self.calls = 0

        def compute(self) -> dict[str, Any]:
            self.calls += 1
            return {"calls": self.calls}

    metric = Counting()

    assert metric.to_dict()["results"] == {"calls": 1}
    assert metric.to_dict()["results"] == {"calls": 2}


def test_to_dict_uses_the_overridden_name_and_description():
    class Named(_Stub):
        def get_name(self) -> str:
            return "overridden"

    assert Named().to_dict()["name"] == "overridden"


# --------------------------------------------------------------------------
# The contract methods refuse rather than silently doing nothing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("collector", "args"),
    [("add_entry", (object(),)), ("add_item", ("anything",)), ("add", (True, False))],
)
def test_collecting_through_a_method_the_metric_does_not_implement_raises(collector, args):
    # A caller has no way to tell add_entry/add_item/add apart except by
    # which one a metric actually implements, so the wrong choice has to
    # raise rather than silently accumulate nothing.
    metric = _Stub()

    with pytest.raises(NotImplementedError, match=collector):
        getattr(metric, collector)(*args)


def test_a_metric_without_a_compute_refuses_to_report():
    # The alternative would be to_dict() serialising "results": null and the
    # modifier silently receiving None where numbers were expected.
    class NoCompute(BaseMetrics):
        pass

    with pytest.raises(NotImplementedError, match="compute"):
        NoCompute("A-Metric", "d").to_dict()


def test_a_subclass_only_has_to_implement_the_collector_it_uses():
    # add_entry/add_item/add refusing by default must not force a metric that
    # only ever collects through one of them to stub out the other two.
    class OnlyItems(BaseMetrics):
        def __init__(self) -> None:
            super().__init__("OnlyItems", "d")
            self.seen: list[Any] = []

        def add_item(self, value: Any) -> None:
            self.seen.append(value)

        def compute(self) -> list[Any]:
            return self.seen

    metric = OnlyItems()
    metric.add_item("a")

    assert metric.to_dict()["results"] == ["a"]
    with pytest.raises(NotImplementedError):
        metric.add(True, True)


# --------------------------------------------------------------------------
# Annotations
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "owner",
    [BaseMetrics, TokenUsageMetrics],
    ids=["m_base", "m_tokens"],
)
def test_the_entry_annotation_names_the_real_entry_class(owner):
    # get_type_hints resolves the annotation to the actual class; comparing
    # against the literal string "Entry" would still pass if the import were
    # removed or the name typo'd elsewhere.
    assert typing.get_type_hints(owner.add_entry)["entry"] is Entry
