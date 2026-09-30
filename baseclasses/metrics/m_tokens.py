from typing import Any

from chain_checker.baseclasses.corpus.entry import Entry
from chain_checker.baseclasses.metrics.m_base import BaseMetrics
from chain_checker.baseclasses.metrics.names import chain_token_usage_metrics


class TokenUsageMetrics(BaseMetrics):
    def __init__(self, name: str = chain_token_usage_metrics) -> None:
        description = (
            "Token usage of the chain itself (not the modifier LLM), summed across "
            "every entry that carries usage. A recalled/cached entry keeps the usage "
            "of the call that first produced it, restored from disk with the "
            "prediction, so these numbers describe what the whole corpus cost to "
            "produce rather than what this run spent on fresh calls - compare epochs "
            "on them, not the wall-clock cost of one run. 'avg_tokens_per_entry' "
            "divides by 'entries_with_usage', not the corpus size, so a run whose "
            "entries predate usage tracking reports zero instead of a division error."
        )
        super().__init__(name, description)
        self._usages: list[dict[str, Any]] = []

    def add_entry(self, entry: Entry) -> None:
        usage = entry.get_model_output().get_token_usage()
        if self._total_of(usage):
            self._usages.append(usage)

    @staticmethod
    def _field(usage: dict[str, Any], key: str) -> int:
        # A token count must be a whole number: bool is an int subtype (so
        # isinstance(True, int) is True) and a fractional float can't be a
        # real token count, so both are treated as absent rather than
        # silently truncated or miscounted.
        value = usage.get(key)
        if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
            return 0
        try:
            return int(value or 0)

        except TypeError, ValueError, OverflowError:
            return 0

    @classmethod
    def _total_of(cls, usage: dict[str, Any]) -> int:
        # Same fallback as ChainAnalyser._extract_usage uses when a call's
        # usage is first captured: some providers/saved predictions omit an
        # explicit total, so it's derived from the two halves instead.
        total = cls._field(usage, "total_tokens")
        if total:
            return total
        return cls._field(usage, "prompt_tokens") + cls._field(usage, "completion_tokens")

    def compute(self) -> dict[str, Any]:
        count = len(self._usages)
        total = sum(self._total_of(u) for u in self._usages)
        return {
            "prompt_tokens": sum(self._field(u, "prompt_tokens") for u in self._usages),
            "completion_tokens": sum(self._field(u, "completion_tokens") for u in self._usages),
            "total_tokens": total,
            "entries_with_usage": count,
            "avg_tokens_per_entry": (total / count) if count else 0.0,
        }
