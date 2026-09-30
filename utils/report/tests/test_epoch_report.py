from chain_checker.baseclasses.metrics.names import (
    accuracy_metrics,
    chain_token_usage_metrics,
    language_metrics,
    modification_metrics,
    negative_predicted_metrics,
    text_length_metrics,
)
from chain_checker.utils.report.epoch_report import ReportGenerator


def _run_metrics(accuracy: float) -> dict:
    return {
        accuracy_metrics: {"accuracy": accuracy, "rules": {}},
        chain_token_usage_metrics: {"total_tokens": 100, "entries_with_usage": 2},
    }


# ---- single run vs. multi-run detection ----


def test_a_single_runs_metrics_dict_is_wrapped_and_rendered_without_a_comparison_section():
    html = ReportGenerator.for_run(_run_metrics(0.8)).render()

    assert html.count('<section class="run">') == 1
    assert 'class="comparison"' not in html


def test_two_or_more_runs_get_their_own_section_plus_a_comparison_section():
    html = ReportGenerator({"run_0": _run_metrics(0.6), "run_1": _run_metrics(0.9)}).render()

    assert html.count('<section class="run">') == 2
    assert 'class="comparison"' in html
    assert "Overall accuracy by run" in html


def test_the_accuracy_comparison_is_scaled_0_to_100_not_against_the_other_runs():
    # A weak run (12%) next to a strong one (96%) must keep its own width -
    # render_bars() would instead stretch it to fill the track relative to
    # the strongest run, which would misread as "almost as good".
    generator = ReportGenerator({"run_0": _run_metrics(0.12), "run_1": _run_metrics(0.96)})

    html = generator._render_comparison()

    assert "width:12.0%" in html
    assert "width:96.0%" in html


def test_generate_writes_the_rendered_html_to_the_given_path(tmp_path):
    out_path = tmp_path / "nested" / "report.html"
    generator = ReportGenerator.for_run(_run_metrics(0.8))

    result_path = generator.generate(str(out_path))

    assert result_path == str(out_path)
    assert out_path.read_text().startswith("<!DOCTYPE html>")


# ---- overall accuracy: missing vs. present ----
# Regression: a run with no Accuracy-Metrics block used to read as 0.0 and
# render a big red "0.0% Overall accuracy" tile, which then also made that
# run look like the worst one in a multi-run comparison.


def test_a_run_without_an_accuracy_block_renders_no_overall_accuracy_tile():
    metrics = {chain_token_usage_metrics: {"total_tokens": 100, "entries_with_usage": 2}}

    html = ReportGenerator.for_run(metrics).render()

    assert "Overall accuracy" not in html
    assert 'class="big-stat-value big-stat-primary"' not in html


def test_a_run_without_an_accuracy_block_is_excluded_from_the_run_comparison():
    html = ReportGenerator({"run_0": {}, "run_1": _run_metrics(0.9)})._render_comparison()

    accuracy_section = html.split("Chain token usage by run")[0]
    assert "run_0" not in accuracy_section
    assert "run_1" in accuracy_section and "width:90.0%" in accuracy_section


# ---- rule-card dispatch, by result shape ----


def test_a_boolean_confusion_matrix_rule_renders_as_a_confusion_card():
    result = {"accuracy": 0.8, "TT": 4, "TF": 1, "FT": 0, "FF": 5}

    html = ReportGenerator._render_rule_card("brand-uppercase", result)

    assert "brand-uppercase" in html
    assert "80%" in html


def test_a_numeric_rule_renders_with_an_mae_badge_and_paired_bars():
    result = {
        "accuracy": 0.9,
        "mae": 0.05,
        "true_scores": [1.0, 0.9],
        "predicted_scores": [0.95, 0.85],
    }

    html = ReportGenerator._render_rule_card("score", result)

    assert "MAE 0.050" in html
    assert "paired-chart" in html


def test_an_agreement_rule_renders_as_matched_mismatched_bars():
    result = {"matched": 1, "mismatched": 1, "total": 2, "accuracy": 0.5}

    html = ReportGenerator._render_rule_card("tonality", result)

    assert "50%" in html
    assert "Matched" in html


def test_an_unrecognized_shape_falls_back_to_a_raw_json_dump():
    html = ReportGenerator._render_rule_card("weird", {"foo": "bar"})

    assert "foo" in html and "bar" in html
    assert 'class="raw"' in html


# ---- run-info chips ----


def test_run_info_chips_render_known_keys_with_friendly_labels():
    generator = ReportGenerator({})

    html = generator._render_run_info(
        {"chain": "template_checklist", "tier": "fast", "model": "gpt-4o-mini"}
    )

    assert "Chain" in html and "template_checklist" in html
    assert "Tier" in html and "fast" in html
    assert "Model" in html and "gpt-4o-mini" in html


def test_run_info_chips_are_empty_for_no_info():
    assert ReportGenerator({})._render_run_info({}) == ""


def test_run_info_chips_ignore_unknown_keys():
    assert ReportGenerator({})._render_run_info({"unexpected": "value"}) == ""


# ---- parsing metrics ----


def test_parsing_metrics_lists_failed_entry_ids_and_the_failure_rate():
    payload = {
        "total": 9,
        "parsed": 7,
        "failed": 2,
        "failure_rate": 2 / 9,
        "failed_ids": ["3", "7"],
    }

    html = ReportGenerator._render_parsing_metrics(payload)

    assert "3" in html and "7" in html
    assert "22%" in html


def test_parsing_metrics_shows_none_when_nothing_failed():
    payload = {"total": 9, "parsed": 9, "failed": 0, "failure_rate": 0.0, "failed_ids": []}

    html = ReportGenerator._render_parsing_metrics(payload)

    assert "none" in html


# ---- mispredictions ----


def test_no_mispredictions_shows_a_reassuring_empty_state():
    html = ReportGenerator({})._render_mispredictions({})

    assert "No mispredictions." in html


def test_mispredicted_entries_render_true_vs_predicted_per_key():
    mispredictions = {
        "case-3": {
            "input": {"text": "hello"},
            "true": {"passed": True},
            "predicted": {"passed": False},
            "matches": {"passed": False},
        }
    }

    html = ReportGenerator({})._render_mispredictions(mispredictions)

    assert "case-3" in html
    assert "mispred-key-diff" in html
    assert "true" in html and "false" in html


def _run_metrics_with_mispredictions(accuracy: float) -> dict:
    metrics = _run_metrics(accuracy)
    metrics[negative_predicted_metrics] = {
        "case-3": {
            "input": {"text": "hello"},
            "true": {"passed": True},
            "predicted": {"passed": False},
            "matches": {"passed": False},
        }
    }
    return metrics


def test_mispredictions_show_by_default_in_a_single_runs_report():
    html = ReportGenerator.for_run(_run_metrics_with_mispredictions(0.8)).render()

    assert "Mispredicted entries" in html
    assert "case-3" in html


def test_show_mispredictions_false_drops_the_section_entirely_not_just_the_entries():
    # Used by runs_overview.html - each run's mispredictions belong on that
    # epoch's own report.html, not repeated per run on the comparison page.
    # Passing show_mispredictions=False must skip the whole section, not
    # fall back to the "No mispredictions." empty state.
    html = ReportGenerator.for_run(
        _run_metrics_with_mispredictions(0.8), show_mispredictions=False
    ).render()

    assert "Mispredicted entries" not in html
    assert "case-3" not in html
    assert "No mispredictions." not in html


def test_show_mispredictions_false_still_renders_everything_else():
    with_mispredictions = ReportGenerator.for_run(_run_metrics_with_mispredictions(0.8)).render()
    without_mispredictions = ReportGenerator.for_run(
        _run_metrics_with_mispredictions(0.8), show_mispredictions=False
    ).render()

    assert "Overall accuracy" in without_mispredictions
    assert len(without_mispredictions) < len(with_mispredictions)


# ---- prompt rendering ----


def _run_metrics_with_prompt(accuracy: float) -> dict:
    metrics = _run_metrics(accuracy)
    metrics["prompt"] = "You review marketing text for brand compliance."
    return metrics


def test_the_prompt_shows_as_an_always_open_card_by_default():
    html = ReportGenerator.for_run(_run_metrics_with_prompt(0.8)).render()

    assert '<div class="card prompt-card">' in html
    assert "<details" not in html
    assert "You review marketing text for brand compliance." in html


def test_collapse_prompt_true_renders_it_behind_a_details_disclosure_instead():
    # Used by runs_overview.html - N runs' full prompts, all expanded by
    # default, is what makes the comparison page hard to scan.
    html = ReportGenerator.for_run(_run_metrics_with_prompt(0.8), collapse_prompt=True).render()

    assert '<details class="prompt-entry">' in html
    assert "<summary>Model prompt</summary>" in html
    assert "You review marketing text for brand compliance." in html
    assert '<div class="card prompt-card">' not in html


def test_collapse_prompt_true_renders_nothing_when_there_is_no_prompt():
    metrics = _run_metrics(0.8)

    html = ReportGenerator.for_run(metrics, collapse_prompt=True).render()

    assert "<details" not in html


# ---- other generic metric-card sections ----


def test_generic_metric_sections_render_only_when_present():
    metrics = _run_metrics(0.8)
    metrics[modification_metrics] = ["swap", "typo", "swap"]
    metrics[text_length_metrics] = [12, 45, 9]
    metrics[language_metrics] = {"en": 2, "de": 1}

    html = ReportGenerator.for_run(metrics).render()

    assert modification_metrics in html
    assert '<div class="histogram">' in html
    assert language_metrics in html


def test_chain_token_section_is_skipped_when_no_entry_ever_reported_usage():
    # Regression: a run whose entries predate usage tracking still reports a
    # full Chain-Token-Usage-Metrics dict with every value at zero, which is
    # truthy as a dict even though there's nothing real to chart.
    metrics = _run_metrics(0.8)
    metrics[chain_token_usage_metrics] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "entries_with_usage": 0,
        "avg_tokens_per_entry": 0.0,
    }

    html = ReportGenerator.for_run(metrics).render()

    assert chain_token_usage_metrics not in html
