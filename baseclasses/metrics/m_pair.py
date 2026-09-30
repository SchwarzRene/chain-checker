from typing import Any

from chain_checker.baseclasses.corpus.label import values_agree
from chain_checker.baseclasses.metrics.m_base import BaseMetrics


class PairMetrics(BaseMetrics):
    """Base for metrics fed one (true, predicted) pair at a time via `add()`
    - `BooleanMetrics`, `FloatMetrics` and `ValueMetrics` each interpret the
    same accumulated pairs differently in their own `compute()`."""

    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description)
        self._true_values: list[Any] = []
        self._pred_values: list[Any] = []

    def add(self, true_value: Any, pred_value: Any) -> None:
        self._true_values.append(true_value)
        self._pred_values.append(pred_value)

    def _agreement(self) -> tuple[int, int]:
        agreed = sum(
            values_agree(true_value, pred_value)
            for true_value, pred_value in zip(self._true_values, self._pred_values, strict=True)
        )
        return agreed, len(self._true_values)
