from __future__ import annotations

from typing import TYPE_CHECKING

from chain_checker.baseclasses.loop.cache import LoopCache
from chain_checker.baseclasses.loop.progress import ProgressTracker
from chain_checker.utils.console import link_print, link_print_warning, next_glyph

if TYPE_CHECKING:
    from chain_checker.baseclasses.chain.model import Model
    from chain_checker.baseclasses.corpus import Corpus, Entry
    from chain_checker.baseclasses.corpus.output import ModelOutput


class ClassificationLoop:
    """Runs every entry in a corpus through a model, one at a time, using
    `LoopCache` to skip entries already predicted under the same prompt and
    config. A single entry raising does not abort the run - it is recorded
    as failed (see `_predict`) and the loop continues with the rest of the
    corpus."""

    def __init__(
        self, run_id: str, corpus: Corpus, model: Model, base_dir: str | None = None
    ) -> None:
        self._run_id = run_id
        self._corpus = corpus
        self._model = model
        self._failed_ids: set[str | int] = set()
        self._last_error: Exception | None = None
        self._cache = LoopCache(run_id, corpus, model, base_dir)

    def get_run_dir(self) -> str:
        return self._cache.get_run_dir()

    def get_failed_ids(self) -> set[str | int]:

        return set(self._failed_ids)

    def loop(self, debug: bool = False) -> None:

        self._failed_ids = set()
        self._last_error = None

        if len(self._corpus) == 0:
            link_print_warning(
                f"(LOOP) WARNING: corpus is empty, nothing to run for run '{self._run_id}'"
            )
            return

        recalled = self._cache.recall()
        self._cache.save_prompt()
        self._cache.save_config()

        total = len(self._corpus)
        tracker = ProgressTracker(total - len(recalled))
        self._print_recall_summary(recalled, total, debug)

        for number, entry in enumerate(self._corpus, start=1):
            position = f"({number}/{total})"
            if str(entry.get_id()) in recalled:
                self._print_recalled(position, entry, debug)
                continue
            self._run_entry(position, entry, tracker, debug)

        self._print_output_shape_suggestion(tracker)

    def _run_entry(
        self, position: str, entry: Entry, tracker: ProgressTracker, debug: bool
    ) -> None:
        if debug:
            link_print(f"(LOOP) {position} entry '{entry.get_id()}' - calling the model...")

        prefix = (
            f"{next_glyph()} (LOOP) {position} entry '{entry.get_id()}' -> "
            f"waiting on chain '{self._model.get_chain_type()}'"
        )
        predicted = self._predict(entry, prefix, tracker)

        if predicted is not None:
            entry.set_model_output(predicted)
            self._cache.save_prediction(entry.get_id(), predicted)
            if debug:
                link_print(
                    f"(LOOP) {position} entry '{entry.get_id()}' - done, saved "
                    f"to {self._cache.get_entry_path(entry.get_id())}"
                )

    def _predict(self, entry: Entry, prefix: str, tracker: ProgressTracker) -> ModelOutput | None:
        # Broad on purpose: a single entry's parsing failure, timeout, or
        # malformed reply must not abort the whole corpus run - it is
        # recorded as failed and the loop moves on.
        predicted, error, _ = self._model.call_with_progress(entry.get_input(), prefix, tracker)
        if error is not None:
            self._print_failure(entry, error)
            self._failed_ids.add(entry.get_id())
            self._last_error = error
            return None
        return predicted

    def _print_recalled(self, position: str, entry: Entry, debug: bool) -> None:
        if not debug:
            return
        link_print(f"(LOOP) {position} entry '{entry.get_id()}' - recalled from disk, skipping")

    def _print_recall_summary(self, recalled: set[str], total: int, debug: bool) -> None:
        if not debug:
            return
        link_print(
            f"(LOOP) run '{self._run_id}' ({self._model.get_type()}/"
            f"{self._model.get_chain_type()}): {len(recalled)}/{total} entries "
            f"recalled from {self._cache.get_entries_dir()}"
        )

    def _print_failure(self, entry: Entry, error: Exception) -> None:
        link_print_warning(
            f"(LOOP) WARNING: run '{self._run_id}' entry '{entry.get_id()}' "
            f"(chain '{self._model.get_chain_type()}', type "
            f"'{self._model.get_type()}') failed: {error} - counting it as an "
            f"empty prediction and continuing with the rest of the corpus."
        )

    def _print_output_shape_suggestion(self, tracker: ProgressTracker) -> None:
        # Only fires on a 100% failure rate. The last error's class is
        # printed rather than asserted as a prompt-shape issue, since a
        # 100% failure rate is just as often an infra problem (missing
        # LiteLLM key, network) as a malformed SYSTEM_PROMPT.
        attempted = tracker.get_done()
        if attempted == 0 or len(self._failed_ids) < attempted:
            return

        link_print(
            f"(LOOP) every attempted entry in run '{self._run_id}' failed; "
            f"last error: {type(self._last_error).__name__}: {self._last_error}"
        )
