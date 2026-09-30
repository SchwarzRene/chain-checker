from typing import Any

from chain_checker.baseclasses.corpus.label import is_number
from chain_checker.baseclasses.metrics.m_pair import PairMetrics


class FloatMetrics(PairMetrics):
    @staticmethod
    def _number_or_none(value: Any) -> float | int | None:
        return value if is_number(value) else None

    def compute(self) -> dict[str, Any]:
        # Only pairs where both sides are numeric feed the MAE - a
        # non-numeric prediction (e.g. a parsing failure) still counts
        # against accuracy via _agreement() below, but averaging it into the
        # MAE would be meaningless.
        pairs = [
            (t, p)
            for t, p in zip(self._true_values, self._pred_values, strict=True)
            if is_number(t) and is_number(p)
        ]

        scored = len(pairs)

        agreed, total = self._agreement()

        return {
            "mae": sum(abs(t - p) for t, p in pairs) / scored if pairs else 0.0,
            "accuracy": agreed / total if total else 0.0,
            "scored": scored,
            "unscored": total - scored,
            # Non-numeric entries become None rather than being dropped, so
            # true_scores/predicted_scores stay index-aligned with each other
            # and a chart can show a gap instead of shifting later points.
            "true_scores": [self._number_or_none(t) for t in self._true_values],
            "predicted_scores": [self._number_or_none(p) for p in self._pred_values],
        }
