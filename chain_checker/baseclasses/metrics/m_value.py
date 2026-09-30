from typing import Any

from chain_checker.baseclasses.metrics.m_pair import PairMetrics


class ValueMetrics(PairMetrics):
    def compute(self) -> dict[str, Any]:
        matched, total = self._agreement()

        return {
            "accuracy": matched / total if total else 0.0,
            "matched": matched,
            "mismatched": total - matched,
            "total": total,
        }
