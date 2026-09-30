import os
from typing import Any

from chain_checker.baseclasses.chain.model import Model
from chain_checker.baseclasses.corpus import Corpus
from chain_checker.baseclasses.loop import ClassificationLoop
from chain_checker.baseclasses.metrics import (
    ModifierMetrics,
    accuracy_metrics,
    chain_token_usage_metrics,
    modifier_run_info,
    modifier_token_usage_metrics,
)
from chain_checker.modifier.model import ModifierModel
from chain_checker.utils.console import link_print
from chain_checker.utils.metrics_file import write_metrics
from chain_checker.utils.report import ReportGenerator
from chain_checker.utils.training_utils import paths


def _run_corpus_pass(
    label: str, corpus_id: str, base_dir: str, corpus: Corpus, model: Model
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    link_print(f"(R)-(MODIFIER) Resetting {label} corpus")
    corpus.reset()

    link_print(f"(R)-(MODIFIER) Initializing the {label} classification loop")
    classification_loop = ClassificationLoop(corpus_id, corpus, model, base_dir=base_dir)

    link_print(f"(R)-(MODIFIER) Looping over the {label} corpus...")
    classification_loop.loop(debug=True)

    link_print(f"(R)-(MODIFIER) Computing {label} metrics")
    metrics = ModifierMetrics(corpus, failed_ids=classification_loop.get_failed_ids()).get_metrics()
    report_data = {item["name"]: item["results"] for item in metrics}

    return report_data, metrics


def _report_epoch_progress(
    kind: str,
    epoch_display: int,
    total_epochs: int,
    run_report_data: dict,
    val_corpus: Corpus | None,
) -> None:
    accuracy = run_report_data.get(accuracy_metrics, {}).get("accuracy")
    chain_tokens = run_report_data.get(chain_token_usage_metrics, {})
    accuracy_label = "train accuracy" if val_corpus is not None else "overall accuracy of"
    link_print(
        f"(R)-(MODIFIER) {kind} {epoch_display}/{total_epochs} done - "
        f"{accuracy_label} {accuracy}, "
        f"chain used {chain_tokens.get('total_tokens', 0)} tokens across "
        f"{chain_tokens.get('entries_with_usage', 0)} call(s) "
        f"(avg {chain_tokens.get('avg_tokens_per_entry', 0):.1f}/entry)"
    )
    if val_corpus is not None:
        val_accuracy = run_report_data["val"].get(accuracy_metrics, {}).get("accuracy")
        val_chain_tokens = run_report_data["val"].get(chain_token_usage_metrics, {})
        link_print(
            f"(R)-(MODIFIER) {kind} {epoch_display}/{total_epochs} val accuracy "
            f"{val_accuracy} - "
            f"chain used {val_chain_tokens.get('total_tokens', 0)} tokens across "
            f"{val_chain_tokens.get('entries_with_usage', 0)} call(s) on the val corpus"
        )


def _rewrite_prompt(
    kind: str,
    epoch_display: int,
    total_epochs: int,
    model: Model,
    modifier: ModifierModel,
    epoch_dir: str,
    run_report_data: dict,
    metrics: list,
) -> None:
    link_print("(R)-(MODIFIER) Generating new prompt")
    new_system_prompt = modifier.modify(model.get_system_prompt(), metrics, epoch_dir)
    model.set_new_system_prompt(new_system_prompt)

    modifier_tokens = modifier.get_last_usage()
    cumulative_tokens = modifier.get_cumulative_usage()
    run_report_data[modifier_token_usage_metrics] = modifier_tokens
    run_report_data[modifier_run_info] = modifier.get_run_info()

    link_print(
        f"(R)-(MODIFIER) {kind} {epoch_display}/{total_epochs} modifier LLM used "
        f"{modifier_tokens.get('total_tokens', 0)} tokens rewriting the prompt "
        f"(cumulative across all epochs so far: {cumulative_tokens.get('total_tokens', 0)})"
    )


def run_epoch(
    epoch: int,
    total_epochs: int,
    corpus: Corpus,
    model: Model,
    modifier: ModifierModel,
    run_dir: str,
    ask_for_rewrite: bool,
    val_corpus: Corpus | None = None,
) -> dict[str, Any]:
    epoch_dir = paths.epoch_dir(run_dir, epoch)
    run_id = os.path.basename(epoch_dir)
    kind = "epoch" if ask_for_rewrite else "final validation pass (no further rewrite)"
    epoch_display = epoch + 1
    run_type = "training epoch" if ask_for_rewrite else "final validation epoch"
    link_print(f"(R)-(MODIFIER) Running {kind} {epoch_display}/{total_epochs}...")
    link_print(f"(R)-(MODIFIER) running {run_type} {epoch_display}:")

    run_report_data, metrics = _run_corpus_pass("train", run_id, run_dir, corpus, model)

    run_report_data["prompt"] = model.get_system_prompt()
    run_report_data["run_info"] = model.get_run_info()
    if not ask_for_rewrite:
        run_report_data["final_validation_only"] = True

    report_bundle = {f"{run_id} (train)" if val_corpus is not None else run_id: run_report_data}

    if val_corpus is not None:
        val_report_data, _ = _run_corpus_pass("val", "val", epoch_dir, val_corpus, model)
        run_report_data["val"] = val_report_data
        report_bundle[f"{run_id} (val)"] = val_report_data

    report = ReportGenerator(report_bundle)
    report_path = report.generate(os.path.join(epoch_dir, "report.html"))
    link_print(f"(R)-(MODIFIER)-(REPORT) Report written to {report_path}")

    _report_epoch_progress(kind, epoch_display, total_epochs, run_report_data, val_corpus)

    if ask_for_rewrite:
        _rewrite_prompt(
            kind, epoch_display, total_epochs, model, modifier, epoch_dir, run_report_data, metrics
        )

    write_metrics(epoch_dir, run_report_data)

    return run_report_data
