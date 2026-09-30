import os

from chain_checker.baseclasses.chain.model import Model
from chain_checker.baseclasses.loop.paths import default_base_dir
from chain_checker.utils.report import ReportGenerator, SummaryReportGenerator
from chain_checker.utils.training_utils.metrics_io import load_epoch_metrics
from chain_checker.utils.training_utils.paths import (
    epoch_number,
    existing_run_numbers,
    run_dir as build_run_dir,
)


def write_summary(run_dir: str) -> str:
    summary_dir = os.path.join(run_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    report = SummaryReportGenerator(load_epoch_metrics(run_dir))
    return report.generate(os.path.join(summary_dir, "report.html"))


def write_runs_overview(model: Model) -> str | None:
    base_dir = default_base_dir(model.get_type(), model.get_chain_type())

    latest_per_run = {}
    for n in existing_run_numbers(base_dir):
        epoch_metrics = load_epoch_metrics(build_run_dir(base_dir, n))
        if epoch_metrics:
            latest_epoch = max(epoch_metrics, key=epoch_number)
            latest_per_run[f"run_{n}"] = epoch_metrics[latest_epoch]

    if len(latest_per_run) < 2:
        return None

    report = ReportGenerator(latest_per_run, show_mispredictions=False, collapse_prompt=True)
    return report.generate(os.path.join(base_dir, "runs_overview.html"))
