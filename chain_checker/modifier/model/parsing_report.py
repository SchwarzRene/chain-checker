from typing import Any


def build_parse_failure_report(
    parsing_results: dict[str, Any], mispredicted_results: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Reduces Parsing-Metrics' failed_ids to {id: {"input": ...}} - just
    enough for the compact report to render. The input is borrowed from the
    matching Negative-Predicted-Metrics entry, since an unparsed reply still
    has a real input recorded against it."""
    mispredicted_by_str_id = {
        str(entry_id): entry for entry_id, entry in mispredicted_results.items()
    }
    return {
        str(entry_id): {"input": mispredicted_by_str_id.get(str(entry_id), {}).get("input", {})}
        for entry_id in (parsing_results.get("failed_ids") or [])
    }
