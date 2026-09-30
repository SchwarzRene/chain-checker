import html as html_
import os
import sys
from typing import Any

from chain_checker.baseclasses.metrics.m_single_value import (
    RulePath,
    iter_rule_results,
    rule_label,
)
from chain_checker.baseclasses.metrics.names import (
    accuracy_metrics,
    chain_token_usage_metrics,
    modifier_run_info,
    modifier_token_usage_metrics,
)
from chain_checker.utils.report import palette
from chain_checker.utils.report.css import shared_css, summary_report_css
from chain_checker.utils.report.hover_script import HOVER_SCRIPT
from chain_checker.utils.report.line_charts import (
    render_line_chart,
    render_stacked_area_chart,
)

_MODIFIER_RUN_INFO_LABELS = {"backend": "Modifier", "app": "App", "tier": "Tier", "model": "Model"}
_CHAIN_RUN_INFO_LABELS = {"chain": "Chain", "tier": "Tier", "model": "Model"}


def _epoch_index(name: str) -> int:
    suffix = name.removeprefix("modification_")
    # Anything that is not modification_N sorts last instead of crashing the page.
    return int(suffix) if suffix.isdigit() else sys.maxsize


class SummaryReportGenerator:
    def __init__(self, epoch_metrics: dict[str, dict[str, Any]]) -> None:
        self._epochs: list[tuple[str, dict[str, Any]]] = sorted(
            epoch_metrics.items(), key=lambda kv: _epoch_index(kv[0])
        )
        # Computed once so every chart walks the same per-epoch breakdown
        # instead of re-parsing Accuracy-Metrics per chart.
        self._epoch_rule_results: list[dict[RulePath, dict[str, Any]]] = [
            dict(iter_rule_results(metrics.get(accuracy_metrics, {})))
            for _, metrics in self._epochs
        ]

        self._chart_id = 0

    def generate(self, output_path: str = "report.html") -> str:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.render())

        return output_path

    def render(self) -> str:
        return self._page(self._render_body() + HOVER_SCRIPT)

    def _page(self, body: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Training Summary</title>
<style>{shared_css() + summary_report_css()}</style>
</head>
<body>
{body}
</body>
</html>"""

    def _next_chart_id(self) -> str:
        self._chart_id += 1
        return f"chart-{self._chart_id}"

    def _render_body(self) -> str:
        if not self._epochs:
            return """
            <section class="section">
              <h1 class="run-title">Training Summary</h1>
              <div class="empty">
                No modification_N/metrics.json found yet for this run - this page fills in as
                epochs complete and save one. If epochs already exist under this run_N but this
                still shows empty, they were made by an older version of trainingLoop.py that
                never wrote metrics.json; a --continue can't reconstruct their trend data, only
                new epochs from here on will appear.
              </div>
            </section>"""

        return (
            f"""
            <section class="section">
              <h1 class="run-title">Training Summary</h1>
              <div class="epoch-count">{len(self._epochs)} epoch(s) completed</div>
              <div class="run-info-groups">
                {self._render_chain_info()}
                {self._render_modifier_info()}
              </div>
              {self._render_headline_stats()}
            </section>"""
            + self._render_best_prompt()
            + self._render_accuracy_chart()
            + self._render_mae_chart()
            + self._render_sub_metric_chart()
            + self._render_chain_token_chart()
            + self._render_modifier_token_chart()
            + self._render_prompt_evolution()
        )

    # ---- headline ----

    def _render_info_row(self, metrics_key: str, labels: dict[str, str], group_label: str) -> str:
        for _, metrics in reversed(self._epochs):
            run_info = metrics.get(metrics_key)
            if run_info:
                chips = "".join(
                    f"""
            <div class="run-info-chip">
              <span class="run-info-key">{html_.escape(labels.get(key, key))}</span>
              <span class="run-info-value">{html_.escape(str(run_info[key]))}</span>
            </div>"""
                    for key in labels
                    if key in run_info
                )
                if not chips:
                    return ""
                return f"""
            <div class="run-info-group">
              <div class="run-info-group-label">{html_.escape(group_label)}</div>
              <div class="run-info">{chips}</div>
            </div>"""
        return ""

    def _render_chain_info(self) -> str:
        return self._render_info_row("run_info", _CHAIN_RUN_INFO_LABELS, "Chain being trained")

    def _render_modifier_info(self) -> str:
        return self._render_info_row(
            modifier_run_info, _MODIFIER_RUN_INFO_LABELS, "Prompt modifier"
        )

    def _epoch_accuracy(self, metrics: dict[str, Any]) -> float | None:
        return metrics.get(accuracy_metrics, {}).get("accuracy")

    def _best_epoch(self) -> tuple[str, dict[str, Any]]:
        # Highest accuracy wins; on a tie, the cheapest prompt (fewest chain
        # tokens) does. Used everywhere "best epoch" is picked so headline
        # stats and the best-prompt callout never disagree. Epochs with no
        # Accuracy-Metrics block (a crashed/incomplete run) are excluded
        # rather than counted as 0.0, so they can never look "worst".
        scored = [
            (name, metrics)
            for name, metrics in self._epochs
            if self._epoch_accuracy(metrics) is not None
        ]
        if not scored:
            return self._epochs[-1]

        return max(
            scored,
            key=lambda kv: (
                self._epoch_accuracy(kv[1]),
                -kv[1].get(chain_token_usage_metrics, {}).get("total_tokens", 0),
            ),
        )

    def _render_headline_stats(self) -> str:
        latest_label, latest_metrics = self._epochs[-1]
        latest_label = latest_label.removeprefix("modification_")
        latest = self._epoch_accuracy(latest_metrics)

        best_label, best_metrics = self._best_epoch()
        best_label = best_label.removeprefix("modification_")
        best = self._epoch_accuracy(best_metrics)

        return f"""
        <div class="big-stats">
          {
            self._render_accuracy_stat(
                latest, f"Latest accuracy (epoch {html_.escape(latest_label)})", primary=True
            )
        }
          {self._render_accuracy_stat(best, f"Best accuracy (epoch {html_.escape(best_label)})")}
        </div>"""

    @staticmethod
    def _render_accuracy_stat(accuracy: float | None, label: str, primary: bool = False) -> str:
        value_class = "big-stat-value big-stat-primary" if primary else "big-stat-value"
        if accuracy is None:
            return f"""
          <div class="big-stat">
            <div class="{value_class}">N/A</div>
            <div class="big-stat-label">{label}</div>
          </div>"""

        color = palette.accuracy_color(accuracy)
        return f"""
          <div class="big-stat">
            <div class="{value_class}" style="color:{color}">{accuracy * 100:.1f}%</div>
            <div class="big-stat-label">{label}</div>
          </div>"""

    # ---- best-prompt callout (only when the run didn't end on its best epoch) ----

    def _render_best_prompt(self) -> str:
        latest_label, latest_metrics = self._epochs[-1]
        best_label, best_metrics = self._best_epoch()

        accuracy = self._epoch_accuracy(best_metrics)
        latest_accuracy = self._epoch_accuracy(latest_metrics)
        # Hide on an exact match, a tie (max() can land on an earlier epoch
        # than the latest one even when their accuracy is equal), or when no
        # epoch has an accuracy value to highlight as "best" at all.
        if accuracy is None or (latest_accuracy is not None and accuracy <= latest_accuracy):
            return ""

        accuracy_color = palette.accuracy_color(accuracy)
        tokens = best_metrics.get(chain_token_usage_metrics, {}).get("total_tokens", 0)
        stats = f"""
        <div class="big-stats">
          <div class="big-stat">
            <div class="big-stat-value big-stat-primary"
                 style="color:{accuracy_color}">{accuracy * 100:.1f}%</div>
            <div class="big-stat-label">Accuracy</div>
          </div>
          <div class="big-stat">
            <div class="big-stat-value">{tokens:,}</div>
            <div class="big-stat-label">Tokens</div>
          </div>
        </div>"""

        latest_tokens = latest_metrics.get(chain_token_usage_metrics, {}).get("total_tokens", 0)
        tokens_delta = tokens - latest_tokens
        accuracy_delta_text = (
            ""
            if latest_accuracy is None
            else f"{(accuracy - latest_accuracy) * 100:+.1f}% accuracy, "
        )
        deltas_block = (
            f'<div class="best-prompt-deltas">{accuracy_delta_text}'
            f"{tokens_delta:+,} tokens vs. the latest epoch</div>"
        )

        epoch_label = best_label.removeprefix("modification_")
        heading = f"Best prompt - epoch {html_.escape(epoch_label)} ({tokens:,} tokens)"
        return f"""
        <section class="section">
          <h2 class="section-title">{heading}</h2>
          <div class="card">
            {stats}
            {deltas_block}
            <pre class="prompt-text">{html_.escape(str(best_metrics.get("prompt", "")))}</pre>
          </div>
        </section>"""

    # ---- charts ----

    def _epoch_labels(self) -> list[str]:
        return [name.removeprefix("modification_") for name, _ in self._epochs]

    def _render_accuracy_chart(self) -> str:
        series: dict[str, list[float | None]] = {
            "Accuracy": [
                None if (a := self._epoch_accuracy(metrics)) is None else a * 100
                for _, metrics in self._epochs
            ]
        }
        chart = render_line_chart(
            self._next_chart_id(),
            series,
            self._epoch_labels(),
            y_domain=(0, 100),
            y_format=lambda v: f"{v:.0f}%",
            show_legend=False,
        )
        return f"""
        <section class="section">
          <h2 class="section-title">Overall accuracy by epoch</h2>
          <div class="card">{chart}</div>
        </section>"""

    def _sub_rule_paths(self) -> list[RulePath]:
        seen: list[RulePath] = []
        for results in self._epoch_rule_results:
            for path in results:
                if path not in seen:
                    seen.append(path)
        return seen

    @staticmethod
    def _metric_at(
        results: dict[RulePath, dict[str, Any]], path: RulePath, key: str
    ) -> float | None:
        return results[path][key] if path in results and key in results[path] else None

    def _is_numeric_rule(self, path: RulePath) -> bool:
        # A numeric ("mae") rule also reports "accuracy" (its within-tolerance
        # match rate), but that's tracked by the MAE chart, not this one.
        return any(
            self._metric_at(results, path, "mae") is not None
            for results in self._epoch_rule_results
        )

    def _render_sub_metric_chart(self) -> str:
        paths = [path for path in self._sub_rule_paths() if not self._is_numeric_rule(path)]
        if not paths:
            return ""

        series = {
            rule_label(path): [
                None if (v := self._metric_at(results, path, "accuracy")) is None else v * 100
                for results in self._epoch_rule_results
            ]
            for path in paths
        }

        chart = render_line_chart(
            self._next_chart_id(),
            series,
            self._epoch_labels(),
            y_domain=(0, 100),
            y_format=lambda v: f"{v:.0f}%",
        )
        return f"""
        <section class="section">
          <h2 class="section-title">Sub-metric accuracy by epoch</h2>
          <div class="card">{chart}</div>
        </section>"""

    def _render_mae_chart(self) -> str:
        series = {}
        for path in self._sub_rule_paths():
            values = [self._metric_at(results, path, "mae") for results in self._epoch_rule_results]
            if any(v is not None for v in values):
                series[rule_label(path)] = values

        if not series:
            return ""

        chart = render_line_chart(self._next_chart_id(), series, self._epoch_labels())
        return f"""
        <section class="section">
          <h2 class="section-title">Score error (MAE) by epoch - lower is better</h2>
          <div class="card">{chart}</div>
        </section>"""

    def _token_usage_layers(
        self, metrics_key: str, epochs: list[tuple[str, dict[str, Any]]] | None = None
    ) -> list[tuple[str, list[float | None]]]:
        epochs = self._epochs if epochs is None else epochs
        prompt = [metrics.get(metrics_key, {}).get("prompt_tokens", 0) for _, metrics in epochs]
        completion = [
            metrics.get(metrics_key, {}).get("completion_tokens", 0) for _, metrics in epochs
        ]
        if not any(prompt) and not any(completion):
            return []
        return [("Prompt", prompt), ("Completion", completion)]

    def _render_chain_token_chart(self) -> str:
        layers = self._token_usage_layers(chain_token_usage_metrics)
        if not layers:
            return ""

        area_chart = render_stacked_area_chart(self._next_chart_id(), layers, self._epoch_labels())
        return f"""
            <section class="section">
              <h2 class="section-title">Chain token cost by epoch</h2>
              <div class="card">{area_chart}</div>
            </section>"""

    def _render_modifier_token_chart(self) -> str:
        # The final epoch never calls the modifier (see `run_epoch`'s
        # `ask_for_rewrite` guard), so it has no Modifier-Token-Usage-Metrics.
        # Treating that as 0 would plot a fake drop instead of just no point.
        modifier_epochs = [
            (name, metrics)
            for name, metrics in self._epochs
            if modifier_token_usage_metrics in metrics
        ]
        layers = self._token_usage_layers(modifier_token_usage_metrics, modifier_epochs)
        if not layers:
            return ""

        labels = [name.removeprefix("modification_") for name, _ in modifier_epochs]
        area_chart = render_stacked_area_chart(self._next_chart_id(), layers, labels)
        return f"""
                <section class="section">
                  <h2 class="section-title">Modifier LLM token cost by epoch</h2>
                  <div class="card">{area_chart}</div>
                </section>"""

    # ---- prompt evolution ----

    def _render_prompt_evolution(self) -> str:
        blocks = "".join(
            self._render_prompt_entry(name, metrics)
            for name, metrics in self._epochs
            if metrics.get("prompt")
        )
        if not blocks:
            return ""

        return f"""
        <section class="section">
          <h2 class="section-title">Prompt evolution</h2>
          {blocks}
        </section>"""

    @staticmethod
    def _render_prompt_entry(name: str, metrics: dict[str, Any]) -> str:
        is_final = bool(metrics.get("final_validation_only"))
        return (
            f"""
            <details class="prompt-entry">
              <summary>Epoch {html_.escape(name.removeprefix("modification_"))}"""
            + (" - final, not rewritten further" if is_final else "")
            + f"""</summary>
              <pre class="prompt-text">{html_.escape(str(metrics.get("prompt", "")))}</pre>
            </details>"""
        )
