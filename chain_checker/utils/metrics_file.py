import json
import os
from typing import Any

METRICS_FILE = "metrics.json"


def serialise_metrics(report_data: dict) -> str:
    return json.dumps(report_data, indent=2, sort_keys=True, default=str)


def read_metrics_file(metrics_path: str) -> dict[str, Any] | None:
    """The parsed metrics.json, or None if it is missing or unreadable."""
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except OSError, json.JSONDecodeError:
        return None


def write_metrics(target_dir: str, report_data: dict) -> None:
    """Writes via a temp file + os.replace so a crash never leaves a truncated metrics.json."""
    metrics_path = os.path.join(target_dir, METRICS_FILE)
    tmp_path = f"{metrics_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(serialise_metrics(report_data))
    os.replace(tmp_path, metrics_path)
