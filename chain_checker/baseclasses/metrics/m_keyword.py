from collections import Counter
from collections.abc import Hashable

from chain_checker.baseclasses.metrics.m_base import BaseMetrics


class KeywordMetrics(BaseMetrics):
    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description)
        self._keys: Counter[Hashable] = Counter()

    def add_item(self, value: Hashable) -> None:
        self._keys[value] += 1

    def compute(self) -> dict[str, int]:
        # Keyed by str(value): distinct raw keys that stringify the same way
        # (e.g. 1 and "1") are merged into one bucket, since the report only
        # ever displays the string form anyway.
        counts: dict[str, int] = {}
        for value, count in self._keys.items():
            counts[str(value)] = counts.get(str(value), 0) + count
        return counts
