from typing import Any

from chain_checker.baseclasses.corpus.entry import Entry
from chain_checker.baseclasses.metrics.m_base import BaseMetrics
from chain_checker.baseclasses.metrics.names import negative_predicted_metrics

# A key the model's output never produced is reported as this literal string,
# not None: None is a legal expected value, so it must stay visibly
# distinguishable from "never produced" wherever it's read back (the report,
# any equality check against a real predicted value).
_MISSING = "<missing>"


class MispredictedMetrics(BaseMetrics):
    def __init__(self, name: str = negative_predicted_metrics) -> None:
        description = (
            "One entry per corpus case that failed at least one checked "
            "key: {entry-id: {'input': ..., 'true': ..., 'predicted': ..., "
            "'matches': {key: bool}}} - 'true'/'predicted' only carry the "
            "keys the case actually checks, 'matches' says which of those "
            "keys passed. A key the model's output never carried is "
            f"reported in 'predicted' as the literal string {_MISSING!r}, "
            "not None - None is a legal expected value, so it must stay "
            "distinguishable from a key that was simply never produced."
        )
        super().__init__(name, description)
        self._entries: dict[str, dict[str, Any]] = {}

    def add_entry(self, entry: Entry) -> None:
        true = entry.get_output()
        pred = entry.get_model_output()
        matches = true.matches(pred)

        if not all(matches.values()):
            pred_data = pred.get()
            self._entries[str(entry.get_id())] = {
                "input": entry.get_input().get(),
                "true": true.get(),
                "predicted": {
                    key: pred_data[key] if key in pred_data else _MISSING for key in true.get_keys()
                },
                "matches": matches,
            }

    def compute(self) -> dict[str, dict[str, Any]]:
        return dict(self._entries)
