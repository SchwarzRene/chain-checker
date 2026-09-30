from chain_checker.baseclasses.metrics.names import (
    accuracy_metrics,
    chain_token_usage_metrics,
    modifier_run_info,
    modifier_token_usage_metrics,
)
from chain_checker.utils.report.summary_report import SummaryReportGenerator


def _epoch(accuracy: float, prompt: str = "p", extra: dict | None = None) -> dict:
    metrics = {accuracy_metrics: {"accuracy": accuracy}, "prompt": prompt}
    if extra:
        metrics.update(extra)
    return metrics


# ---- empty state ----


def test_no_epochs_yet_shows_a_placeholder_instead_of_a_blank_page():
    assert "No modification_N/metrics.json found yet" in SummaryReportGenerator({}).render()


# ---- malformed epoch keys ----


def test_a_non_modification_n_epoch_key_sorts_last_instead_of_crashing_the_page():
    # Regression: int(name.removeprefix("modification_")) blew up on any key
    # that wasn't exactly "modification_<N>", taking down the whole page.
    SummaryReportGenerator({"modification_0": {}, "final": {}}).render()


# ---- modifier info chip row ----
# Regression coverage for a real bug: this row used to be keyed on
# "chain"/"tier"/"model" (the *chain's* run-info shape) while the modifier
# actually reports "backend"/"app"/"tier"/"model" - so "Modifier"/"App" never
# rendered at all, silently.


def test_a_litellm_run_shows_modifier_app_tier_and_model_chips():
    epochs = {
        "modification_0": _epoch(
            0.5,
            extra={
                modifier_run_info: {
                    "backend": "litellm",
                    "app": "tonality",
                    "tier": "fast",
                    "model": "gpt-4o-mini",
                }
            },
        )
    }

    html = SummaryReportGenerator(epochs)._render_modifier_info()

    assert "Modifier" in html and "litellm" in html
    assert "App" in html and "tonality" in html
    assert "Tier" in html and "fast" in html
    assert "Model" in html and "gpt-4o-mini" in html


def test_an_ollama_run_shows_only_the_chips_it_actually_reports():
    epochs = {
        "modification_0": _epoch(
            0.5, extra={modifier_run_info: {"backend": "ollama", "model": "qwen3.5:4b"}}
        )
    }

    html = SummaryReportGenerator(epochs)._render_modifier_info()

    assert "Modifier" in html and "ollama" in html
    assert "qwen3.5:4b" in html
    assert "App" not in html
    assert "Tier" not in html


def test_modifier_info_is_empty_when_no_epoch_ever_reported_it():
    html = SummaryReportGenerator({"modification_0": _epoch(0.5)})._render_modifier_info()

    assert html == ""


# ---- chain info chip row ----
# The training summary used to show only the modifier's own info, never
# which chain/tier/model the run itself is actually training.


def test_chain_info_shows_the_chains_own_type_tier_and_model():
    epochs = {
        "modification_0": _epoch(
            0.5,
            extra={
                "run_info": {
                    "chain": "real_template_checklist",
                    "tier": "fast",
                    "model": "fast",
                }
            },
        )
    }

    html = SummaryReportGenerator(epochs)._render_chain_info()

    assert "Chain" in html and "real_template_checklist" in html
    assert "Tier" in html and "fast" in html
    assert "Model" in html


def test_chain_info_is_empty_when_no_epoch_ever_reported_it():
    html = SummaryReportGenerator({"modification_0": _epoch(0.5)})._render_chain_info()

    assert html == ""


def test_chain_info_and_modifier_info_are_independent_rows():
    epochs = {
        "modification_0": _epoch(
            0.5,
            extra={
                "run_info": {"chain": "real_template_checklist", "tier": "fast", "model": "fast"},
                modifier_run_info: {"backend": "litellm", "app": "tonality", "tier": "fast"},
            },
        )
    }
    generator = SummaryReportGenerator(epochs)

    chain_html = generator._render_chain_info()
    modifier_html = generator._render_modifier_info()

    assert "real_template_checklist" in chain_html
    assert "real_template_checklist" not in modifier_html
    assert "litellm" in modifier_html
    assert "litellm" not in chain_html


def test_chain_info_and_modifier_info_are_each_wrapped_in_a_labeled_group():
    # A bare row of Tier/Model chips reads as ambiguous once there are two
    # rows with the same key names - each gets its own boxed group with a
    # caption so it's clear which subject a chip belongs to.
    epochs = {
        "modification_0": _epoch(
            0.5,
            extra={
                "run_info": {"chain": "real_template_checklist", "tier": "fast", "model": "fast"},
                modifier_run_info: {"backend": "litellm", "app": "tonality", "tier": "fast"},
            },
        )
    }
    generator = SummaryReportGenerator(epochs)

    chain_html = generator._render_chain_info()
    modifier_html = generator._render_modifier_info()

    assert '<div class="run-info-group">' in chain_html
    assert "Chain being trained" in chain_html
    assert '<div class="run-info-group">' in modifier_html
    assert "Prompt modifier" in modifier_html


# ---- headline stats ----


def test_headline_stats_report_latest_and_best_epoch_accuracy():
    epochs = {
        "modification_0": _epoch(0.5),
        "modification_1": _epoch(0.9),
        "modification_2": _epoch(0.7),
    }

    html = SummaryReportGenerator(epochs)._render_headline_stats()

    assert "70.0%" in html and "Latest accuracy (epoch 2)" in html
    assert "90.0%" in html and "Best accuracy (epoch 1)" in html


# ---- missing accuracy block (a crashed/incomplete epoch) ----
# Regression: an epoch with no Accuracy-Metrics block used to read as 0.0,
# which rendered a fake "0.0%" tile and made that epoch look like the worst
# one - both in the headline and when picking the "best" epoch.


def test_best_epoch_skips_an_epoch_with_no_accuracy_block_instead_of_treating_it_as_zero():
    epochs = {"modification_0": _epoch(0.5), "modification_1": {"prompt": "crashed"}}

    best_label, _ = SummaryReportGenerator(epochs)._best_epoch()

    assert best_label == "modification_0"


def test_headline_stats_show_n_a_for_an_epoch_with_no_accuracy_block():
    epochs = {"modification_0": _epoch(0.9), "modification_1": {"prompt": "crashed"}}

    html = SummaryReportGenerator(epochs)._render_headline_stats()

    assert "N/A" in html and "Latest accuracy (epoch 1)" in html
    assert "90.0%" in html and "Best accuracy (epoch 0)" in html
    assert '<div class="big-stat-value big-stat-primary">N/A</div>' in html


# ---- best-prompt callout ----


def test_best_prompt_callout_is_hidden_when_the_latest_epoch_is_also_the_best():
    epochs = {"modification_0": _epoch(0.5), "modification_1": _epoch(0.9)}

    assert SummaryReportGenerator(epochs)._render_best_prompt() == ""


def test_best_prompt_callout_is_hidden_on_a_tie_not_just_an_exact_match():
    epochs = {"modification_0": _epoch(0.8), "modification_1": _epoch(0.8)}

    assert SummaryReportGenerator(epochs)._render_best_prompt() == ""


def test_best_prompt_callout_appears_with_deltas_against_the_latest_epoch():
    epochs = {
        "modification_0": _epoch(
            0.9, prompt="peak prompt", extra={chain_token_usage_metrics: {"total_tokens": 2000}}
        ),
        "modification_1": _epoch(
            0.7, prompt="latest prompt", extra={chain_token_usage_metrics: {"total_tokens": 1500}}
        ),
    }

    html = SummaryReportGenerator(epochs)._render_best_prompt()

    assert "Best prompt - epoch 0 (2,000 tokens)" in html
    assert "+20.0% accuracy" in html
    assert "+500 tokens vs. the latest epoch" in html
    assert "peak prompt" in html


def test_render_includes_the_best_prompt_callout_when_the_run_didnt_end_on_its_best_epoch():
    # Regression: _render_best_prompt() was never wired into _render_body(),
    # so the tests above called the private method directly and stayed green
    # while render() itself never actually showed the callout.
    epochs = {"modification_0": _epoch(0.9, prompt="peak prompt"), "modification_1": _epoch(0.7)}

    html = SummaryReportGenerator(epochs).render()

    assert "Best prompt - epoch 0" in html


def test_best_prompt_callout_drops_the_accuracy_delta_when_the_latest_epoch_has_no_accuracy_block():
    epochs = {
        "modification_0": _epoch(0.9, prompt="peak prompt"),
        "modification_1": {"prompt": "crashed"},
    }

    html = SummaryReportGenerator(epochs)._render_best_prompt()

    assert "Best prompt - epoch 0" in html
    assert "% accuracy" not in html
    assert "vs. the latest epoch" in html


# ---- accuracy / sub-metric / MAE charts ----


def test_accuracy_chart_plots_every_epoch_as_a_percentage():
    epochs = {"modification_0": _epoch(0.5), "modification_1": _epoch(0.75)}

    html = SummaryReportGenerator(epochs)._render_accuracy_chart()

    assert "Overall accuracy by epoch" in html
    assert "chart-svg" in html


def test_accuracy_chart_skips_an_epoch_with_no_accuracy_block_instead_of_plotting_zero():
    epochs = {"modification_0": _epoch(0.8), "modification_1": {"prompt": "crashed"}}

    html = SummaryReportGenerator(epochs)._render_accuracy_chart()

    assert '"title": "Epoch 1", "rows": []' in html


def test_sub_metric_chart_is_empty_without_any_rule_breakdown():
    assert SummaryReportGenerator({"modification_0": _epoch(0.5)})._render_sub_metric_chart() == ""


def test_sub_metric_chart_tracks_each_rule_across_epochs():
    def epoch_with_rule(accuracy: float) -> dict:
        return {
            accuracy_metrics: {
                "accuracy": accuracy,
                "rules": {"brand-uppercase": {"results": {"accuracy": accuracy}}},
            },
            "prompt": "p",
        }

    epochs = {"modification_0": epoch_with_rule(0.5), "modification_1": epoch_with_rule(0.9)}

    html = SummaryReportGenerator(epochs)._render_sub_metric_chart()

    assert "brand-uppercase" in html
    assert "Sub-metric accuracy by epoch" in html


def test_sub_metric_chart_excludes_numeric_mae_rules_shown_in_the_mae_chart_instead():
    # Regression: a numeric rule like "score" reports "accuracy" too (its
    # within-tolerance match rate), which used to leak it into this chart
    # alongside real boolean rules even though it already gets its own MAE
    # chart below.
    epochs = {
        "modification_0": {
            accuracy_metrics: {
                "accuracy": 0.5,
                "rules": {
                    "passed": {"results": {"accuracy": 1.0, "TT": 1, "TF": 0, "FT": 0, "FF": 0}},
                    "score": {"results": {"accuracy": 0.0, "mae": 0.7}},
                },
            },
            "prompt": "p",
        }
    }

    html = SummaryReportGenerator(epochs)._render_sub_metric_chart()

    assert "passed" in html
    assert "score" not in html


def test_sub_metric_chart_is_empty_when_every_rule_is_numeric():
    epochs = {
        "modification_0": {
            accuracy_metrics: {
                "accuracy": 0.5,
                "rules": {"score": {"results": {"accuracy": 0.0, "mae": 0.7}}},
            },
            "prompt": "p",
        }
    }

    assert SummaryReportGenerator(epochs)._render_sub_metric_chart() == ""


def test_mae_chart_only_includes_rules_that_actually_report_an_mae():
    epochs = {
        "modification_0": {
            accuracy_metrics: {
                "accuracy": 0.5,
                "rules": {
                    "score": {"results": {"accuracy": 0.5, "mae": 0.2}},
                    "tonality": {"results": {"accuracy": 0.5}},
                },
            },
            "prompt": "p",
        }
    }

    html = SummaryReportGenerator(epochs)._render_mae_chart()

    assert "score" in html
    assert "tonality" not in html


# ---- token-cost charts ----


def test_the_chain_token_chart_renders_once_any_epoch_reports_chain_tokens():
    epochs = {
        "modification_0": _epoch(
            0.5, extra={chain_token_usage_metrics: {"prompt_tokens": 500, "completion_tokens": 100}}
        )
    }

    assert "chart-svg" in SummaryReportGenerator(epochs)._render_chain_token_chart()


def test_the_chain_token_chart_is_empty_without_any_reported_tokens():
    epochs = {"modification_0": _epoch(0.5)}

    assert SummaryReportGenerator(epochs)._render_chain_token_chart() == ""


def test_the_modifier_token_chart_skips_the_final_epoch_that_never_ran_the_modifier():
    epochs = {
        "modification_0": _epoch(
            0.5,
            extra={
                modifier_token_usage_metrics: {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                }
            },
        ),
        "modification_1": _epoch(
            0.6,
            extra={
                modifier_token_usage_metrics: {
                    "prompt_tokens": 120,
                    "completion_tokens": 60,
                    "total_tokens": 180,
                }
            },
        ),
        "modification_2": _epoch(0.65, extra={"final_validation_only": True}),
    }

    html = SummaryReportGenerator(epochs)._render_modifier_token_chart()

    assert '"title": "Epoch 0"' in html
    assert '"title": "Epoch 1"' in html
    assert '"title": "Epoch 2"' not in html


def test_the_modifier_token_chart_is_empty_when_no_epoch_ever_ran_the_modifier():
    epochs = {"modification_0": _epoch(0.5, extra={"final_validation_only": True})}

    assert SummaryReportGenerator(epochs)._render_modifier_token_chart() == ""


# ---- prompt evolution ----


def test_prompt_evolution_marks_the_final_validation_only_epoch():
    epochs = {
        "modification_0": _epoch(0.5, prompt="v0"),
        "modification_1": _epoch(0.6, prompt="v1", extra={"final_validation_only": True}),
    }

    html = SummaryReportGenerator(epochs)._render_prompt_evolution()

    assert "Epoch 0" in html and "v0" in html
    assert "Epoch 1 - final, not rewritten further" in html
    assert "v1" in html


def test_prompt_evolution_is_empty_when_no_epoch_has_a_prompt():
    epochs = {"modification_0": {accuracy_metrics: {"accuracy": 0.5}}}

    assert SummaryReportGenerator(epochs)._render_prompt_evolution() == ""


# ---- generate() ----


def test_generate_writes_the_rendered_html_to_the_given_path(tmp_path):
    out_path = tmp_path / "nested" / "report.html"
    generator = SummaryReportGenerator({"modification_0": _epoch(0.8)})

    result_path = generator.generate(str(out_path))

    assert result_path == str(out_path)
    assert out_path.read_text().startswith("<!DOCTYPE html>")
