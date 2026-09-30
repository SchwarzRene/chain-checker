from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from chain_checker.baseclasses.loop.paths import (
    RunPaths,
    default_base_dir,
    entry_id_from_filename,
)
from chain_checker.baseclasses.loop.prediction_file import (
    parse_prediction,
    render_prediction,
)
from chain_checker.utils.console import link_print, link_print_warning

if TYPE_CHECKING:
    from chain_checker.baseclasses.chain.model import Model
    from chain_checker.baseclasses.corpus import Corpus, Entry
    from chain_checker.baseclasses.corpus.output import ModelOutput


def current_prompt(model: Model) -> str:
    try:
        return model.get_system_prompt()
    except AttributeError:
        return ""


def saved_prompt_matches(run_dir: str, prompt: str) -> bool:
    prompt_file = RunPaths.for_run_dir(run_dir).prompt_file()
    if not os.path.isfile(prompt_file):
        return False

    with open(prompt_file, "r") as f:
        return f.read() == prompt


def saved_config_matches(run_dir: str, config: dict[str, Any]) -> bool:
    config_file = RunPaths.for_run_dir(run_dir).config_file()
    if not os.path.isfile(config_file):
        return False

    try:
        with open(config_file, "r") as f:
            return json.load(f) == config
    except json.JSONDecodeError:
        return False


def saved_run_matches(run_dir: str, prompt: str, config: dict[str, Any]) -> bool:
    return saved_prompt_matches(run_dir, prompt) and saved_config_matches(run_dir, config)


class LoopCache:
    """Persists each entry's prediction to disk under the run directory, so a
    resumed/`--continue`d run doesn't re-call the model for entries it has
    already answered. The cache is only trusted (`recall()`) when the saved
    prompt and chain config exactly match the current model - otherwise the
    old predictions no longer describe what the model would say now, and are
    discarded rather than served."""

    def __init__(
        self, run_id: str, corpus: Corpus, model: Model, base_dir: str | None = None
    ) -> None:
        self._corpus = corpus
        self._model = model
        self._paths = RunPaths(
            run_id,
            base_dir
            if base_dir is not None
            else default_base_dir(model.get_type(), model.get_chain_type()),
        )

    def get_run_dir(self) -> str:
        return self._paths.run_dir()

    def get_entries_dir(self) -> str:
        return self._paths.entries_dir()

    def get_entry_path(self, entry_id: str | int) -> str:
        return self._paths.entry_file(entry_id)

    def get_failure_path(self, entry_id: str | int) -> str:
        return self._paths.failure_file(entry_id)

    def save_prompt(self) -> None:

        os.makedirs(self._paths.run_dir(), exist_ok=True)
        with open(self._paths.prompt_file(), "w", encoding="utf-8") as f:
            f.write(self._current_prompt())

    def save_config(self) -> None:
        os.makedirs(self._paths.run_dir(), exist_ok=True)
        with open(self._paths.config_file(), "w", encoding="utf-8") as f:
            json.dump(self._model.get_config(), f, indent=2, sort_keys=True)

    def save_prediction(self, entry_id: str | int, predicted: ModelOutput) -> None:
        os.makedirs(self._paths.entries_dir(), exist_ok=True)
        with open(self._paths.entry_file(entry_id), "w", encoding="utf-8") as f:
            f.write(render_prediction(predicted))

    def save_failure(self, entry_id: str | int, predicted: ModelOutput) -> None:
        """Keeps what a failed case actually sent and got back, for reading -
        deliberately not under entries/, where recall() would serve its empty
        output as a real prediction and the case would never be retried."""
        os.makedirs(self._paths.failures_dir(), exist_ok=True)
        with open(self._paths.failure_file(entry_id), "w", encoding="utf-8") as f:
            f.write(render_prediction(predicted))

    def _current_prompt(self) -> str:
        return current_prompt(self._model)

    def recall(self) -> set[str]:
        # Prompt or config changed since these predictions were cached: they
        # no longer reflect what the current model would produce, so wipe
        # them rather than let them linger unused on disk.
        if not self._saved_matches_current():
            self._discard_stale_entries()
            return set()

        if not os.path.isdir(self._paths.entries_dir()):
            return set()

        entries_by_id = self._entries_by_id()
        recalled = set()
        for filename in sorted(os.listdir(self._paths.entries_dir())):
            entry_id = entry_id_from_filename(filename)
            if entry_id is None or entry_id not in entries_by_id:
                continue

            predicted = self._read_prediction(filename)
            if predicted is None:
                continue

            entries_by_id[entry_id].set_model_output(predicted)
            recalled.add(entry_id)

        return recalled

    def _read_prediction(self, filename: str) -> ModelOutput | None:
        path = os.path.join(self._paths.entries_dir(), filename)
        try:
            with open(path, "r") as f:
                return parse_prediction(f.read(), path)
        except (OSError, ValueError) as e:
            link_print_warning(
                f"(LOOP) WARNING: could not recall '{path}', re-running this entry instead ({e})"
            )
            return None

    def _saved_matches_current(self) -> bool:
        return saved_run_matches(
            self._paths.run_dir(), self._current_prompt(), self._model.get_config()
        )

    def _discard_stale_entries(self) -> None:

        entries_dir = self._paths.entries_dir()
        if not os.path.isdir(entries_dir):
            return

        # Only touch files that look like saved predictions (id + .txt) - the
        # directory shouldn't hold anything else, but this avoids removing an
        # unrelated file if it ever does.
        stale = [
            filename
            for filename in os.listdir(entries_dir)
            if entry_id_from_filename(filename) is not None
        ]
        if not stale:
            return

        link_print(
            f"(LOOP) the {len(stale)} cached prediction(s) in {entries_dir} "
            f"were made with a different prompt or chain config - discarding "
            f"them and re-running every entry"
        )
        for filename in stale:
            os.remove(os.path.join(entries_dir, filename))

    def _entries_by_id(self) -> dict[str, Entry]:

        return {str(entry.get_id()): entry for entry in self._corpus}
