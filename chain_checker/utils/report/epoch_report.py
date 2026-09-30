import html as html_
import json
import os
from typing import Any

from chain_checker.baseclasses.metrics.m_single_value import (
    iter_rule_results,
    rule_label,
)
from chain_checker.baseclasses.metrics.names import (
    accuracy_metrics,
    chain_token_usage_metrics,
    language_metrics,
    modification_metrics,
    modifier_token_usage_metrics,
    negative_predicted_metrics,
    parsing_metrics,
    text_length_metrics,
)
from chain_checker.utils.report import palette
from chain_checker.utils.report.bar_charts import (
    render_accuracy_bars,
    render_agreement,
    render_bars,
    render_confusion,
    render_histogram,
    render_paired_bars,
)
from chain_checker.utils.report.css import epoch_report_css, shared_css
from chain_checker.utils.report.formatting import format_value

# Metric keys rendered as a plain bars/histogram card in the grid, in display
# order. Everything else in a run's metrics dict belongs to the summary
# report or gets its own dedicated section below, not the generic grid.
_LANGUAGE_KEY = language_metrics
_MODIFICATION_KEY = modification_metrics
_TEXT_LENGTH_KEY = text_length_metrics
_PARSING_KEY = parsing_metrics
_CHAIN_TOKENS_KEY = chain_token_usage_metrics

_RUN_INFO_LABELS = {"chain": "Chain", "tier": "Tier", "model": "Model"}

_CHAIN_TOKEN_LABELS = (
    ("prompt_tokens", "Prompt"),
    ("completion_tokens", "Completion"),
    ("total_tokens", "Total"),
    ("entries_with_usage", "Entries"),
    ("avg_tokens_per_entry", "Avg/entry"),
)


class ReportGenerator:
    def __init__(
        self,
        runs: dict[str, Any],
        show_mispredictions: bool = True,
        collapse_prompt: bool = False,
    ) -> None:
        self._runs = runs
        # `write_runs_overview()` compares many runs at a glance - each run's
        # full mispredicted-entries list belongs on its own report.html instead.
        self._show_mispredictions = show_mispredictions
        # Same reasoning for the prompt: N runs' prompts all expanded by default
        # make the overview hard to scan, so it collapses behind a <details>.
        self._collapse_prompt = collapse_prompt

    @classmethod
    def for_run(cls, metrics: dict[str, Any], **kwargs: Any) -> "ReportGenerator":
        return cls({"run": metrics}, **kwargs)

    def generate(self, output_path: str = "report.html") -> str:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.render())

        return output_path

    def render(self) -> str:
        body = self._render_comparison() + "".join(
            self._render_run(run_id, metrics) for run_id, metrics in self._runs.items()
        )
        return self._page(body)

    def _page(self, body: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Metrics Report</title>
<style>{shared_css() + epoch_report_css()}</style>
</head>
<body>
{body}
</body>
</html>"""

    # ---- multi-run comparison (top of a runs_overview.html) ----

    def _render_comparison(self) -> str:
        if len(self._runs) < 2:
            return ""

        accuracy_by_run = {}
        chain_tokens_by_run = {}
        modifier_tokens_by_run = {}
        for run_id, metrics in self._runs.items():
            accuracy = metrics.get(accuracy_metrics, {}).get("accuracy")
            if accuracy is not None:
                accuracy_by_run[run_id] = round(accuracy * 100, 1)

            chain_tokens = metrics.get(_CHAIN_TOKENS_KEY, {}).get("total_tokens")
            if chain_tokens:
                chain_tokens_by_run[run_id] = chain_tokens

            modifier_tokens = metrics.get(modifier_token_usage_metrics, {}).get("total_tokens")
            if modifier_tokens:
                modifier_tokens_by_run[run_id] = modifier_tokens

        sections = [
            ("Overall accuracy by run", render_accuracy_bars(accuracy_by_run)),
            ("Chain token usage by run", render_bars(chain_tokens_by_run)),
        ]
        if modifier_tokens_by_run:
            sections.append(
                ("Modifier LLM token usage by run", render_bars(modifier_tokens_by_run))
            )

        return "".join(
            f"""
            <section class="comparison">
              <h2 class="run-title">{title}</h2>
              {chart}
            </section>"""
            for title, chart in sections
        )

    # ---- a single run's report ----

    def _render_run(self, run_id: str, metrics: dict[str, Any]) -> str:
        run_info = metrics.get("run_info", {})
        prompt = metrics.get("prompt")
        accuracy_payload = metrics.get(accuracy_metrics, {})
        overall_accuracy = accuracy_payload.get("accuracy")
        overall_score_mae = self._extract_overall_score_mae(accuracy_payload)
        mispredictions = (
            metrics.get(negative_predicted_metrics) if self._show_mispredictions else None
        )

        sections = []
        if accuracy_metrics in metrics:
            sections.append(self._render_accuracy_metrics(accuracy_payload))
        sections.extend(self._render_generic_sections(metrics))

        return f"""
        <section class="run">
          <h1 class="run-title">{html_.escape(str(run_id))}</h1>
          {self._render_run_info(run_info)}
          {self._render_big_stats(overall_accuracy, overall_score_mae)}
          <div class="grid">
            {"".join(sections)}
          </div>
          {self._render_prompt(prompt)}
          {self._render_mispredictions(mispredictions) if mispredictions is not None else ""}
        </section>"""

    @staticmethod
    def _extract_overall_score_mae(accuracy_payload: dict[str, Any]) -> float | None:
        maes = [r["mae"] for _, r in iter_rule_results(accuracy_payload) if "mae" in r]
        return maes[0] if len(maes) == 1 else None

    def _render_run_info(self, run_info: dict[str, Any]) -> str:
        if not run_info:
            return ""

        chips = "".join(
            f"""
            <div class="run-info-chip">
              <span class="run-info-key">{html_.escape(_RUN_INFO_LABELS.get(key, key))}</span>
              <span class="run-info-value">{html_.escape(str(run_info[key]))}</span>
            </div>"""
            for key in _RUN_INFO_LABELS
            if key in run_info
        )
        return f'<div class="run-info">{chips}</div>' if chips else ""

    def _render_big_stats(self, accuracy: float | None, overall_score_mae: float | None) -> str:
        stats = []
        if accuracy is not None:
            color = palette.accuracy_color(accuracy)
            stats.append(f"""
            <div class="big-stat">
              <div class="big-stat-value big-stat-primary"
                   style="color:{color}">{accuracy * 100:.1f}%</div>
              <div class="big-stat-label">Overall accuracy</div>
            </div>""")
        if overall_score_mae is not None:
            stats.append(f"""
            <div class="big-stat">
              <div class="big-stat-value">{overall_score_mae:.3f}</div>
              <div class="big-stat-label">Overall score MAE</div>
            </div>""")

        return f'<div class="big-stats">{"".join(stats)}</div>' if stats else ""

    def _render_prompt(self, prompt: str | None) -> str:
        if not prompt:
            return ""

        if self._collapse_prompt:
            return f"""
        <details class="prompt-entry">
          <summary>Model prompt</summary>
          <pre class="prompt-text">{html_.escape(str(prompt))}</pre>
        </details>"""

        return f"""
        <div class="card prompt-card">
          <div class="card-header"><span class="card-title">Model prompt</span></div>
          <pre class="prompt-text">{html_.escape(str(prompt))}</pre>
        </div>"""

    # ---- Accuracy-Metrics: one card per rule, dispatched by result shape ----

    def _render_accuracy_metrics(self, payload: dict[str, Any]) -> str:
        return "".join(
            self._render_rule_card(rule_label(path), result)
            for path, result in iter_rule_results(payload)
        )

    @staticmethod
    def _render_rule_card(name: str, result: dict[str, Any]) -> str:
        if {"TT", "TF", "FT", "FF"} <= result.keys():
            badge = f"{result.get('accuracy', 0.0) * 100:.0f}%"
            return render_confusion(
                name, badge, result["TT"], result["TF"], result["FT"], result["FF"]
            )

        if "mae" in result:
            true_scores = result.get("true_scores", result.get("true-scores", []))
            pred_scores = result.get("predicted_scores", result.get("predicted-scores", []))
            return f"""
        <div class="card">
          <div class="card-header">
            <span class="card-title">{html_.escape(name)}</span>
            <span class="card-badge">MAE {result["mae"]:.3f}</span>
          </div>
          {render_paired_bars(true_scores, pred_scores)}
        </div>"""

        if {"matched", "mismatched", "total"} <= result.keys():
            return render_agreement(name, result)

        return f"""
        <div class="card">
          <div class="card-header"><span class="card-title">{html_.escape(name)}</span></div>
          <div class="raw">{html_.escape(json.dumps(result, indent=2, default=str))}</div>
        </div>"""

    # ---- the other whitelisted per-run metric cards ----

    def _render_generic_sections(self, metrics: dict[str, Any]) -> list[str]:
        sections = []

        if metrics.get(_PARSING_KEY):
            sections.append(self._render_parsing_metrics(metrics[_PARSING_KEY]))

        # Gate on entries_with_usage, not dict truthiness: a run whose entries
        # predate usage tracking still reports a full but all-zero dict, which
        # would otherwise render as a chart of empty bars instead of no card.
        if metrics.get(_CHAIN_TOKENS_KEY, {}).get("entries_with_usage"):
            counts = {
                label: metrics[_CHAIN_TOKENS_KEY][key]
                for key, label in _CHAIN_TOKEN_LABELS
                if key in metrics[_CHAIN_TOKENS_KEY]
            }
            sections.append(self._render_bar_card(_CHAIN_TOKENS_KEY, counts))

        if metrics.get(_MODIFICATION_KEY):
            sections.append(
                self._render_bar_card(_MODIFICATION_KEY, self._tally(metrics[_MODIFICATION_KEY]))
            )

        if metrics.get(_TEXT_LENGTH_KEY):
            sections.append(
                self._render_histogram_card(_TEXT_LENGTH_KEY, metrics[_TEXT_LENGTH_KEY])
            )

        if metrics.get(_LANGUAGE_KEY):
            sections.append(self._render_bar_card(_LANGUAGE_KEY, metrics[_LANGUAGE_KEY]))

        return sections

    @staticmethod
    def _tally(values: list[str]) -> dict[str, float]:
        counts: dict[str, float] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return counts

    @staticmethod
    def _render_bar_card(name: str, counts: dict[str, float]) -> str:
        return f"""
        <div class="card">
          <div class="card-header"><span class="card-title">{html_.escape(name)}</span></div>
          {render_bars(counts)}
        </div>"""

    @staticmethod
    def _render_histogram_card(name: str, values: list[float]) -> str:
        return f"""
        <div class="card">
          <div class="card-header"><span class="card-title">{html_.escape(name)}</span></div>
          {render_histogram(values)}
        </div>"""

    @staticmethod
    def _render_parsing_metrics(payload: dict[str, Any]) -> str:
        counts = {"Parsed": payload.get("parsed", 0), "Failed": payload.get("failed", 0)}
        failed_ids = payload.get("failed_ids", [])
        ids_block = (
            f'<div class="raw">{html_.escape(", ".join(str(i) for i in failed_ids))}</div>'
            if failed_ids
            else '<div class="empty">none</div>'
        )
        return f"""
        <div class="card">
          <div class="card-header">
            <span class="card-title">{_PARSING_KEY}</span>
            <span class="card-badge">{payload.get("failure_rate", 0.0) * 100:.0f}% failed</span>
          </div>
          {render_bars(counts)}
          <div class="card-header"><span class="card-title">Failed entry ids</span></div>
          {ids_block}
        </div>"""

    # ---- mispredicted entries ----

    def _render_mispredictions(self, mispredictions: dict[str, Any]) -> str:
        if not mispredictions:
            return """
            <div class="card mispred-section">
              <div class="card-header"><span class="card-title">Mispredicted entries</span></div>
              <div class="empty">No mispredictions.</div>
            </div>"""

        cards = "".join(
            self._render_mispred_card(entry_id, entry) for entry_id, entry in mispredictions.items()
        )
        return f"""
        <div class="mispred-section">
          <h2 class="run-title">Mispredicted entries ({len(mispredictions)})</h2>
          <div class="mispred-list">
            {cards}
          </div>
        </div>"""

    def _render_mispred_card(self, entry_id: str, entry: dict[str, Any]) -> str:
        matches = entry.get("matches", {})
        true_data = entry.get("true", {})
        predicted_data = entry.get("predicted", {})
        rule_blocks = "".join(
            self._render_mispred_row(key, true_data.get(key), predicted_data.get(key), matched)
            for key, matched in matches.items()
        )
        entry_input = html_.escape(json.dumps(entry.get("input", {}), indent=2, default=str))
        return f"""
        <div class="card mispred-card">
          <div class="card-header">
            <span class="card-title">{html_.escape(str(entry_id))}</span>
          </div>
          <pre class="mispred-text">{entry_input}</pre>
          <div class="mispred-rows">
            {rule_blocks}
          </div>
        </div>"""

    def _render_mispred_row(self, key: str, true_value: Any, pred_value: Any, matched: bool) -> str:
        row_class = "" if matched else " mispred-key-diff"
        return f"""
        <div class="mispred-key{row_class}">
          <div class="mispred-key-name">{html_.escape(str(key))}</div>
          <div class="mispred-key-sides">
            {self._render_mispred_side("true", true_value)}
            {self._render_mispred_side("predicted", pred_value)}
          </div>
        </div>"""

    @staticmethod
    def _render_mispred_side(label: str, value: Any) -> str:
        color = f' style="color:{palette.FALSE_COLOR}"' if label == "predicted" else ""
        return f"""
        <div class="mispred-side">
          <div class="mispred-side-label">{label}</div>
          <div class="mispred-side-value"{color}>{html_.escape(format_value(value))}</div>
        </div>"""
