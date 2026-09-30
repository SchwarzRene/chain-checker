import os
from collections.abc import Iterable
from typing import Any

from chain_checker.baseclasses.chain.model import Model as Chain
from chain_checker.baseclasses.corpus import Corpus
from chain_checker.baseclasses.loop.cache import current_prompt, saved_run_matches
from chain_checker.baseclasses.loop.paths import (
    default_base_dir,
    numbered_dirs,
    slugify_tier,
)
from chain_checker.baseclasses.metrics import ModifierMetrics
from chain_checker.utils.metrics_file import write_metrics
from chain_checker.utils.report import ReportGenerator


def _next_check_dir(base_dir: str, tier: str | None = None) -> str:
    existing = sorted(numbered_dirs(base_dir, "check"))
    next_number = (existing[-1] + 1) if existing else 0
    suffix = f"_{slugify_tier(tier)}" if tier else ""
    return os.path.join(base_dir, f"check_{next_number}{suffix}")


def _build_report_data(
    model: Chain, corpus: Corpus, failed_ids: Iterable[str | int] = frozenset()
) -> dict[str, Any]:
    metrics = ModifierMetrics(corpus, failed_ids=failed_ids).get_metrics()
    report_data = {item["name"]: item["results"] for item in metrics}
    report_data["run_info"] = model.get_run_info()

    try:
        report_data["prompt"] = model.get_system_prompt()
    except AttributeError:
        # Not every chain checker.py runs is a leaf chain with a rewritable
        # SYSTEM_PROMPT (see README.md's chain section) - the report just
        # omits the prompt card for one that isn't.
        pass

    return report_data


def next_check_dir(model: Chain) -> str:
    base_dir = default_base_dir(model.get_type(), model.get_chain_type())
    return _next_check_dir(base_dir, model.get_tier())


def existing_check_dirs(model: Chain) -> list[str]:
    base_dir = default_base_dir(model.get_type(), model.get_chain_type())
    names = numbered_dirs(base_dir, "check")
    return [os.path.join(base_dir, names[number]) for number in sorted(names, reverse=True)]


def resumable_check_dir(model: Chain) -> str | None:
    prompt = current_prompt(model)
    config = model.get_config()
    for candidate in existing_check_dirs(model):
        if saved_run_matches(candidate, prompt, config):
            return candidate
    return None


def write_report(
    model: Chain,
    corpus: Corpus,
    check_dir: str | None = None,
    failed_ids: Iterable[str | int] = frozenset(),
) -> str:
    if check_dir is None:
        check_dir = next_check_dir(model)
    os.makedirs(check_dir, exist_ok=True)

    report_data = _build_report_data(model, corpus, failed_ids)

    write_metrics(check_dir, report_data)

    report = ReportGenerator.for_run(report_data)
    return report.generate(os.path.join(check_dir, "report.html"))
