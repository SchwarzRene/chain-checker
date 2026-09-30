import html as html_
from collections import Counter
from typing import Any

from chain_checker.utils.report import palette
from chain_checker.utils.report.formatting import format_value


def render_bars(
    counts: dict[str, float], suffix: str = "", bar_color: str = palette.BAR_COLOR
) -> str:
    if not counts:
        return '<div class="empty">no data</div>'

    max_value = max(counts.values()) or 1
    rows = []
    for label, value in counts.items():
        width = max(0.0, min(value / max_value, 1.0)) * 100
        rows.append(f"""
        <div class="bar-row">
          <div class="bar-label" title="{html_.escape(str(label))}">{html_.escape(str(label))}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{width:.1f}%;--bar-color:{bar_color}"></div>
          </div>
          <div class="bar-value">{html_.escape(format_value(value))}{suffix}</div>
        </div>""")

    return f'<div class="bars">{"".join(rows)}</div>'


def render_accuracy_bars(accuracy_by_run: dict[str, float]) -> str:
    """Like `render_bars`, but scaled against a fixed 0-100 axis and colored by
    `palette.accuracy_color` so one run's bar is comparable on its own, not just
    relative to the others."""
    if not accuracy_by_run:
        return '<div class="empty">no data</div>'

    rows = []
    for label, value in accuracy_by_run.items():
        width = max(0.0, min(value, 100.0))
        bar_color = palette.accuracy_color(value / 100)
        rows.append(f"""
        <div class="bar-row">
          <div class="bar-label" title="{html_.escape(str(label))}">{html_.escape(str(label))}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{width:.1f}%;--bar-color:{bar_color}"></div>
          </div>
          <div class="bar-value">{html_.escape(format_value(value))}%</div>
        </div>""")

    return f'<div class="bars">{"".join(rows)}</div>'


def render_confusion(name: str, badge: str, tt: int, tf: int, ft: int, ff: int) -> str:
    def cell(value: int, is_diag: bool) -> str:
        color = palette.TRUE_COLOR if is_diag else palette.FALSE_COLOR
        return f"""
            <div class="cm-cell" style="background:{color}1a;border-color:{color}55">
              <div class="cm-value" style="color:{color}">{value}</div>
            </div>"""

    return f"""
    <div class="card">
      <div class="card-header">
        <span class="card-title">{html_.escape(name)}</span>
        <span class="card-badge">{badge}</span>
      </div>
      <div class="confusion">
        <div class="cm-axis"></div>
        <div class="cm-axis">Predicted&nbsp;true</div>
        <div class="cm-axis">Predicted&nbsp;false</div>
        <div class="cm-axis">Actual&nbsp;true</div>
        {cell(tt, True)}
        {cell(tf, False)}
        <div class="cm-axis">Actual&nbsp;false</div>
        {cell(ft, False)}
        {cell(ff, True)}
      </div>
    </div>"""


def render_histogram(values: list[float], bar_color: str = palette.BAR_COLOR) -> str:
    if not values:
        return '<div class="empty">no data</div>'

    counts = Counter(values)
    max_count = max(counts.values())
    cols = []
    for label in sorted(counts):
        count = counts[label]
        height = (count / max_count) * 100
        cols.append(f"""
        <div class="hist-col">
          <div class="hist-count">{count}</div>
          <div class="hist-bar" style="height:{height:.1f}%;--bar-color:{bar_color}"></div>
          <div class="hist-label">{html_.escape(str(label))}</div>
        </div>""")

    return f'<div class="histogram">{"".join(cols)}</div>'


def render_paired_bars(true_values: list[float | None], pred_values: list[float | None]) -> str:
    if not true_values:
        return '<div class="empty">no data</div>'

    numeric = [v for v in true_values + pred_values if v is not None]
    max_value = max(numeric) if numeric else 1
    max_value = max_value or 1

    def bar_col(value: float | None, color: str) -> str:
        if value is None:
            return f"""
            <div class="paired-bar-col">
              <div class="paired-value">-</div>
              <div class="paired-bar" style="height:0%;background:{color};opacity:0.25"></div>
            </div>"""

        height = (value / max_value) * 100
        return f"""
        <div class="paired-bar-col">
          <div class="paired-value">{value:g}</div>
          <div class="paired-bar" style="height:{height:.1f}%;background:{color}"></div>
        </div>"""

    def legend_item(color: str, label: str) -> str:
        return (
            f'<span class="legend-item"><span class="legend-dot" '
            f'style="background:{color}"></span>{label}</span>'
        )

    groups = []
    for i, (true_value, pred_value) in enumerate(zip(true_values, pred_values, strict=True)):
        groups.append(f"""
            <div class="paired-group">
              <div class="paired-bars">
                {bar_col(true_value, palette.TRUE_SCORE_COLOR)}
                {bar_col(pred_value, palette.PRED_SCORE_COLOR)}
              </div>
              <div class="paired-label">#{i + 1}</div>
            </div>""")

    return f"""
        <div class="paired-legend">
          {legend_item(palette.TRUE_SCORE_COLOR, "true")}
          {legend_item(palette.PRED_SCORE_COLOR, "predicted")}
        </div>
        <div class="paired-chart">{"".join(groups)}</div>"""


def render_agreement(name: str, results: dict[str, Any]) -> str:
    total = results.get("total", 0)
    if not total:
        return f"""
    <div class="card">
      <div class="card-header"><span class="card-title">{html_.escape(name)}</span></div>
      <div class="empty">no data</div>
    </div>"""

    matched = results.get("matched", 0)
    mismatched = results.get("mismatched", total - matched)
    accuracy = results.get("accuracy", matched / total)

    rows = [
        ("Matched", matched, palette.TRUE_COLOR),
        ("Mismatched", mismatched, palette.FALSE_COLOR),
    ]

    def bar_row(label: str, value: int, color: str) -> str:
        pct = (value / total) * 100
        return f"""
        <div class="bar-row">
          <div class="bar-label" title="{label}">{label}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{pct:.1f}%;--bar-color:{color}"></div>
          </div>
          <div class="bar-value">{value}</div>
        </div>"""

    bar_rows = "".join(bar_row(label, value, color) for label, value, color in rows)

    return f"""
    <div class="card">
      <div class="card-header">
        <span class="card-title">{html_.escape(name)}</span>
        <span class="card-badge">{accuracy * 100:.0f}% matched</span>
      </div>
      <div class="bars">{bar_rows}</div>
    </div>"""
