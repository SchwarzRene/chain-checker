from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import yaml

from chain_checker.baseclasses.corpus.entry import Entry
from chain_checker.baseclasses.corpus.input import Input
from chain_checker.baseclasses.corpus.label import (
    Kind,
    Label,
    kind_conflict_message,
    value_kind,
)
from chain_checker.baseclasses.corpus.output_empty import EmptyModelOutput
from chain_checker.utils.errors import fail


class Corpus:
    """Loads the fixed set of test cases from a corpus .yaml into `Entry`
    objects. All shape and consistency checks (required keys, id
    collisions, a key's value type staying consistent across cases, ...)
    happen up front in `load()`, so a malformed corpus fails fast with a
    specific message instead of surfacing as a confusing error deep inside
    a metrics computation."""

    def __init__(self) -> None:
        self._entries: dict[str | int, Entry] = {}

    def __str__(self) -> str:
        return f"Corpus\n   Length: {len(self)}\n"

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Entry]:
        return iter(list(self._entries.values()))

    def add_entry(self, entry: Entry) -> None:
        if entry.get_id() in self._entries:
            raise ValueError(f"Corpus already has an entry with id {entry.get_id()!r}")
        self._entries[entry.get_id()] = entry

    def get_by_idx(self, idx: int) -> Entry:
        try:
            return list(self._entries.values())[idx]
        except IndexError:
            raise IndexError(
                f"Corpus index {idx} is out of range - the corpus holds {len(self)} entries."
            ) from None

    def get_by_id(self, entry_id: str | int) -> Entry:
        try:
            return self._entries[entry_id]
        except KeyError:
            known = list(self._entries)
            preview = known[:10]
            raise KeyError(
                f"Corpus has no entry with id {entry_id!r} - it holds "
                f"{len(known)} entries with id(s) {preview}"
                f"{' ...' if len(known) > 10 else ''}."
            ) from None

    def _check_file_data(self, data: Any, path: str) -> list[Any]:
        if not isinstance(data, dict) or "cases" not in data:
            fail(
                f"Corpus file '{path}' has no top-level 'cases' list.\n"
                f"  A corpus .yaml needs a top-level 'cases: [...]' key."
            )

        cases = data["cases"]
        if not isinstance(cases, list):
            fail(
                f"Corpus file '{path}': 'cases' has the wrong type.\n"
                f"  Expected: a list.\n"
                f"  Found: {type(cases).__name__}."
            )

        if not cases:
            fail(
                f"Corpus file '{path}': 'cases' is empty.\n"
                f"  A corpus needs at least one case to be worth running."
            )

        return cases

    def _check_case(self, case: Any, i: int, path: str, seen: dict[str | int, Entry]) -> None:
        if not isinstance(case, dict):
            fail(
                f"Corpus file '{path}': case #{i} has the wrong type.\n"
                f"  Expected: a mapping.\n"
                f"  Found: {type(case).__name__}."
            )

        missing = [key for key in ("id", "input", "output") if key not in case]
        if missing:
            id_hint = f" (id {case['id']!r})" if "id" in case else ""
            fail(
                f"Corpus file '{path}': case #{i}{id_hint} is missing required key(s).\n"
                f"  Missing: {missing}\n"
                f"  Found key(s): {sorted(case, key=str)}\n"
                f"  Every case needs exactly these top-level keys: id, "
                f"input, output (info is optional)."
            )

        self._check_case_id(case["id"], i, path, seen)

        if case.get("info") is not None and not isinstance(case["info"], dict):
            fail(
                f"Corpus file '{path}': case '{case['id']}' has an invalid 'info'.\n"
                f"  Found: {type(case['info']).__name__}.\n"
                f"  'info' is optional, but if present it must be a mapping."
            )

        if not isinstance(case["input"], dict):
            fail(
                f"Corpus file '{path}': case '{case['id']}' has an invalid 'input'.\n"
                f"  Found: {type(case['input']).__name__}.\n"
                f"  'input' must be a mapping of field name to value, "
                f"matching the chain's InputSchema."
            )

        if not case["input"]:
            fail(
                f"Corpus file '{path}': case '{case['id']}' has an empty 'input'.\n"
                f"  A case needs at least one input field for the chain to "
                f"run on, matching the chain's InputSchema."
            )

        if not isinstance(case["output"], dict):
            fail(
                f"Corpus file '{path}': case '{case['id']}' has an invalid 'output'.\n"
                f"  Found: {type(case['output']).__name__}.\n"
                f"  'output' must be a mapping of the keys it wants "
                f"checked to their expected values."
            )

        if not case["output"]:
            fail(
                f"Corpus file '{path}': case '{case['id']}' has an empty 'output'.\n"
                f"  A case needs at least one checked key, otherwise it "
                f"can never actually fail."
            )

        for block in ("input", "output"):
            unquoted = [key for key in case[block] if not isinstance(key, str)]
            if unquoted:
                fail(
                    f"Corpus file '{path}': case '{case['id']}' has "
                    f"non-string {block} field name(s).\n"
                    f"  Field(s): {unquoted}\n"
                    f"  A field name is read back as text everywhere it is "
                    f"used, from the chain's InputSchema to the rule names "
                    f"in the report.\n"
                    f"  Quote them in the .yaml."
                )

    @staticmethod
    def _check_key_kinds(entries: dict[str | int, Entry], path: str) -> None:
        first_seen: dict[str, tuple[Kind, str | int, Any]] = {}

        for entry in entries.values():
            output = entry.get_output()
            for key in output.get_keys():
                value = output.get_value(key)
                kind = value_kind(value)

                if key not in first_seen:
                    first_seen[key] = (kind, entry.get_id(), value)
                    continue

                established, first_id, first_value = first_seen[key]
                if established != kind:
                    fail(
                        kind_conflict_message(
                            key, first_id, first_value, entry.get_id(), value, path
                        )
                    )

    @staticmethod
    def _check_case_id(entry_id: Any, i: int, path: str, seen: dict[str | int, Entry]) -> None:
        if not isinstance(entry_id, (str, int)) or isinstance(entry_id, bool):
            fail(
                f"Corpus file '{path}': case #{i}'s id has the wrong type.\n"
                f"  Found: {entry_id!r} ({type(entry_id).__name__}).\n"
                f"  A case id must be a string or an integer, not a "
                f"float, a bool, a list or a mapping."
            )

        if entry_id in seen:
            fail(
                f"Corpus file '{path}': duplicate case id.\n"
                f"  id: '{entry_id}'\n"
                f"  Each case needs a unique id."
            )

        text = str(entry_id)

        clashing = [other for other in seen if str(other) == text]
        if clashing:
            fail(
                f"Corpus file '{path}': case #{i}'s id collides with an "
                f"earlier id as text.\n"
                f"  This case's id: {entry_id!r}\n"
                f"  Earlier id: {clashing[0]!r}\n"
                f"  Both stringify to '{text}'. The run cache keys a "
                f"prediction by str(id), so the two cases would share one "
                f"cached prediction - give them ids that differ as text too."
            )

        separators = {s for s in ("/", "\\", "\0", os.sep, os.altsep) if s is not None}
        if not text or text in (".", "..") or any(sep in text for sep in separators):
            fail(
                f"Corpus file '{path}': case #{i}'s id cannot be used as a file name.\n"
                f"  id: {entry_id!r}\n"
                f"  The run cache writes each prediction to '<id>.txt' "
                f"inside the run directory, so an id must be non-empty and "
                f"free of path separators."
            )

    def load(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except OSError as e:
            fail(f"Could not open corpus file '{path}'.\n  Reason: {e.strerror or e}")

        except (yaml.YAMLError, UnicodeDecodeError) as e:
            fail(f"Corpus file '{path}' is not valid YAML.\n  Reason: {e}")

        cases = self._check_file_data(data, path)

        entries: dict[str | int, Entry] = {}

        for i, case in enumerate(cases, start=1):
            self._check_case(case, i, path, entries)

            entries[case["id"]] = Entry(
                case["id"],
                Input(case["input"]),
                Label(case["output"]),
                case.get("info") or {},
                EmptyModelOutput(),
            )

        self._check_key_kinds(entries, path)

        self._entries = entries

    def reset(self) -> None:
        # Clears predictions from a previous pass so a fresh classification
        # loop can't accidentally score against stale output from an earlier
        # epoch or corpus run.
        for entry in self:
            entry.set_model_output(EmptyModelOutput())
