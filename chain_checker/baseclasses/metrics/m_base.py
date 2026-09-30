from typing import Any

from chain_checker.baseclasses.corpus.entry import Entry


class BaseMetrics:
    """Base for every metric collector. A subclass is fed through exactly
    one of `add_entry`, `add_item` or `add` (whichever matches what it
    collects) and must implement `compute()`; `to_dict()` is the shared
    shape used to report a metric's name, description and computed
    results."""

    def __init__(self, name: str, description: str) -> None:
        self._name: str = name
        self._description: str = description

    def get_name(self) -> str:
        return self._name

    def get_description(self) -> str:
        return self._description

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.get_name(),
            "description": self.get_description(),
            "results": self.compute(),
        }

    def _not_collected(self, collector: str) -> NotImplementedError:
        return NotImplementedError(
            f"{type(self).__name__} does not collect through {collector}() - "
            f"check how it is fed: add_entry, add_item, add, or a collector "
            f"of its own."
        )

    def add_entry(self, entry: Entry) -> None:
        raise self._not_collected("add_entry")

    def add_item(self, value: Any) -> None:
        raise self._not_collected("add_item")

    def add(self, true_value: Any, pred_value: Any) -> None:
        raise self._not_collected("add")

    def compute(self) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} must implement compute() - to_dict() "
            f"reports whatever it returns as this metric's 'results'."
        )
