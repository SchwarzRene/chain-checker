from collections.abc import Iterable
from typing import Any

from chain_checker.baseclasses.corpus import Corpus
from chain_checker.baseclasses.metrics.m_keyword import KeywordMetrics
from chain_checker.baseclasses.metrics.m_list import ListMetrics
from chain_checker.baseclasses.metrics.m_mispredicted import MispredictedMetrics
from chain_checker.baseclasses.metrics.m_parsing import ParsingMetrics
from chain_checker.baseclasses.metrics.m_single_value import SingleValueMetrics
from chain_checker.baseclasses.metrics.m_tokens import TokenUsageMetrics
from chain_checker.baseclasses.metrics.names import (
    accuracy_metrics,
    labeller_metrics,
    language_metrics,
    modification_metrics,
    text_length_metrics,
)


def _countable(value: Any) -> bool:
    # KeywordMetrics tallies values in a Counter, which requires hashable
    # keys; guard against metadata that came through as something unhashable
    # (e.g. a list) so it's skipped instead of raising, and treat None as
    # "no such metadata" rather than a value worth counting.
    if value is None:
        return False
    try:
        hash(value)
    except TypeError:
        return False
    return True


class ModifierMetrics:
    """Runs every entry in the corpus through the full set of metric
    collectors in a single sweep, then returns their combined `to_dict()`
    results. This is the one place all per-entry metrics (accuracy,
    mispredictions, token usage, parsing failures, corpus metadata) are
    assembled for a report."""

    def __init__(self, corpus: Corpus, failed_ids: Iterable[str | int] = frozenset()) -> None:
        self._corpus = corpus
        self._failed_ids: frozenset[str | int] = frozenset(failed_ids)

    def get_metrics(self) -> list[dict[str, Any]]:
        accuracy = SingleValueMetrics(accuracy_metrics)
        mispredicted = MispredictedMetrics()
        token_usage = TokenUsageMetrics()
        parsing = ParsingMetrics()
        modifications = ListMetrics(
            modification_metrics,
            "The modification type of every corpus entry, one value per entry, "
            "e.g. ['swap', 'typo', 'swap'].",
        )
        text_lengths = ListMetrics(
            text_length_metrics,
            "The word count of every entry's input text, one value per entry, e.g. [12, 45, 9].",
        )
        languages = KeywordMetrics(
            language_metrics,
            "How often each language occurs across the corpus entries, e.g. {'en': 10, 'de': 3}.",
        )
        labellers = KeywordMetrics(
            labeller_metrics,
            "How many texts each labeller labelled, e.g. {'alice': 5, 'bob': 4}.",
        )

        entry_ids: set[str | int] = set()

        for entry in self._corpus:
            entry_ids.add(entry.get_id())
            accuracy.add_entry(entry)
            mispredicted.add_entry(entry)
            token_usage.add_entry(entry)

            modification = entry.get_info("modification")

            if modification is not None:
                modifications.add_item(modification)

            language = entry.get_info("language")
            if _countable(language):
                languages.add_item(language)

            labeller = entry.get_info("labeller")
            if _countable(labeller):
                labellers.add_item(labeller)

            text = entry.get_input().get_value("text", None)
            if isinstance(text, str):
                text_lengths.add_item(len(text.split()))

        parsing.set(len(self._corpus), self._failed_ids & entry_ids)

        return [
            accuracy.to_dict(),
            mispredicted.to_dict(),
            token_usage.to_dict(),
            parsing.to_dict(),
            modifications.to_dict(),
            text_lengths.to_dict(),
            languages.to_dict(),
            labellers.to_dict(),
        ]
