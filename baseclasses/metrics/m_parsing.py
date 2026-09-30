from collections.abc import Iterable
from typing import Any

from chain_checker.baseclasses.metrics.m_base import BaseMetrics
from chain_checker.baseclasses.metrics.names import parsing_metrics

_EntryId = str | int


class ParsingMetrics(BaseMetrics):
    def __init__(self, name: str = parsing_metrics) -> None:
        description = (
            "How many of this run's entries produced a usable, parseable "
            "prediction ('parsed') vs. how many raised while being "
            "classified ('failed') - usually the model's reply not "
            "matching the chain's required output shape. 'failed_ids' "
            "names exactly which entries these were, so a mismatch on one "
            "of them (see Negative-Predicted-Metrics) can be read as a "
            "parsing failure, not a wrong judgment."
        )
        super().__init__(name, description)
        self._total = 0
        self._failed_ids: list[str] = []

    def set(self, total: int, failed_ids: Iterable[_EntryId]) -> None:
        self._total = total
        self._failed_ids = sorted({str(entry_id) for entry_id in failed_ids})

    def compute(self) -> dict[str, Any]:
        failed = len(self._failed_ids)
        return {
            "total": self._total,
            "parsed": self._total - failed,
            "failed": failed,
            "failure_rate": (failed / self._total) if self._total else 0.0,
            "failed_ids": list(self._failed_ids),
        }
