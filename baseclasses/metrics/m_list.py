from typing import Any

from chain_checker.baseclasses.metrics.m_base import BaseMetrics


class ListMetrics(BaseMetrics):
    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description)
        self._items: list[Any] = []

    def add_item(self, value: Any) -> None:
        self._items.append(value)

    def compute(self) -> list[Any]:
        return list(self._items)
