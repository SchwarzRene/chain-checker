from typing import Any

from chain_checker.baseclasses.corpus.label import values_agree
from chain_checker.baseclasses.metrics.m_pair import PairMetrics


class BooleanMetrics(PairMetrics):
    def compute(self) -> dict[str, Any]:
        tt = tf = ft = ff = 0
        for true_value, pred_value in zip(self._true_values, self._pred_values, strict=True):
            agrees = values_agree(true_value, pred_value)
            # `is True`, not truthiness: values reaching a BooleanMetrics are
            # already known to be actual bools (see value_kind's "bool" kind),
            # so this only needs to pick which of the two rows to tally.
            if true_value is True:
                if agrees:
                    tt += 1
                else:
                    tf += 1
            else:
                if agrees:
                    ff += 1
                else:
                    ft += 1

        total = len(self._true_values)
        accuracy = (tt + ff) / total if total else 0.0

        return {
            "accuracy": accuracy,
            "TT": tt,
            "TF": tf,
            "FT": ft,
            "FF": ff,
        }
