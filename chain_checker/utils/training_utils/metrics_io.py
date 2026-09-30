import os
from typing import Any

from chain_checker.baseclasses.metrics import modifier_run_info
from chain_checker.utils.console import link_print_warning
from chain_checker.utils.metrics_file import read_metrics_file
from chain_checker.utils.training_utils.paths import epoch_names

_NON_METRIC_KEYS = ("prompt", "run_info", "val", "final_validation_only", modifier_run_info)


def load_epoch_metrics(run_dir: str) -> dict[str, Any]:
    all_metrics = {}
    for name in epoch_names(run_dir):
        metrics_path = os.path.join(run_dir, name, "metrics.json")
        if not os.path.isfile(metrics_path):
            continue

        metrics = read_metrics_file(metrics_path)
        if metrics is None:
            link_print_warning(
                f"(R)-(MODIFIER) WARNING: Skipping unreadable {metrics_path} - "
                f"delete it to rerun that epoch."
            )
            continue
        all_metrics[name] = metrics
    return all_metrics


def load_previous_metrics(epoch_dir: str) -> dict[str, Any]:
    return read_metrics_file(os.path.join(epoch_dir, "metrics.json")) or {}


def reconstruct_metrics_list(run_report_data: dict) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": "(historical run - see this epoch's own report.html)",
            "results": results,
        }
        for name, results in run_report_data.items()
        if name not in _NON_METRIC_KEYS
    ]
