from collections.abc import Mapping
from typing import Any

from chain_checker.modifier.model.compact_format.scalars import format_scalar


def _input_dedup_key(input_data: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((k, repr(v)) for k, v in (input_data or {}).items()))


def _format_input(lines: list[str], input_data: dict[str, Any] | None) -> None:
    header_fields, body_blocks = [], []
    for key, value in (input_data or {}).items():
        if isinstance(value, str) and (len(value) > 80 or "\n" in value):
            body_blocks.append((key, value))
        else:
            header_fields.append(f"{key}={format_scalar(value)}")

    if header_fields:
        lines[0] += " (" + ", ".join(header_fields) + ")"
    for key, value in body_blocks:
        lines.append(value if len(body_blocks) == 1 else f"{key}:\n{value}")


def _format_entry_block(
    entry_id: str,
    input_data: dict[str, Any] | None,
    mismatches: dict[str, Any] | None,
    same_input_as: str | None,
    shown_in_run: int | None = None,
) -> str:
    lines = [f"### {entry_id}"]

    if shown_in_run is not None:
        # Printed in full in an earlier run already - the text never changes
        # between runs, so repeating it would only grow the instruction.
        lines[0] += f" (input: see Run-{shown_in_run})"
    elif same_input_as is not None:
        lines.append(f"(same input as {same_input_as})")
    else:
        _format_input(lines, input_data)

    if mismatches:
        lines.append("-- mismatches --")
        for field, diff in mismatches.items():
            lines.append(
                f"{field}: {format_scalar(diff.get('true'))}→{format_scalar(diff.get('predicted'))}"
            )

    return "\n".join(lines)


def _format_entries(
    entries: dict[str, dict[str, Any]],
    empty_note: str,
    shown_in: Mapping[str, int] | None = None,
) -> str:
    if not entries:
        return empty_note

    shown_in = shown_in or {}
    blocks: list[str] = []

    # Several ids (e.g. one per checked rule) can share one underlying case;
    # print that case's input once, and reference it from every id after the
    # first rather than repeat it verbatim per id. An empty input is never a
    # dedup match: it means "no input recorded", not "confirmed identical to
    # every other entry with nothing recorded", so it must not claim a
    # shared input those entries were never actually compared on.
    first_id_by_input: dict[tuple[tuple[str, str], ...], str] = {}
    for entry_id, entry in entries.items():
        input_data = entry.get("input") or {}
        key = _input_dedup_key(input_data) if input_data else None
        first_id = first_id_by_input.get(key) if key is not None else None
        if key is not None and first_id is None:
            first_id_by_input[key] = entry_id
        blocks.append(
            _format_entry_block(
                entry_id,
                input_data,
                entry.get("mismatches"),
                same_input_as=first_id,
                shown_in_run=shown_in.get(entry_id),
            )
        )
    return "\n\n".join(blocks)


def format_false_examples_report(
    entries: dict[str, dict[str, Any]], shown_in: Mapping[str, int] | None = None
) -> str:
    """`shown_in` maps an id to the earlier run whose report already printed
    its input in full; those ids get a one-line reference instead."""
    return _format_entries(
        entries, "(no mispredicted entries this run - every case already passed)", shown_in
    )


def format_parse_failure_report(
    entries: dict[str, dict[str, Any]], shown_in: Mapping[str, int] | None = None
) -> str:
    return _format_entries(
        entries,
        "(no parsing failures this run - every entry produced a scoreable answer)",
        shown_in,
    )
