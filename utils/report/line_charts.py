import html as html_
import json
from collections.abc import Callable
from typing import Any

from chain_checker.utils.report import palette
from chain_checker.utils.report.formatting import format_metric_value, nice_step


def embed_hover_data(
    chart_id: str, xs: list[float], top: float, bottom: float, tooltips: list[dict[str, Any]]
) -> str:
    payload = {"xs": xs, "top": top, "bottom": bottom, "tooltips": tooltips}

    raw = json.dumps(payload, default=str).replace("</", "<\\/")
    return f'<script type="application/json" data-chart-points="{chart_id}">{raw}</script>'


# Shared by both chart kinds below: x-axis epoch ticks and the dot-plus-name
# legend markup.


def _x_axis_ticks_svg(x_labels: list[str], x_at: Callable[[int], float], height: int) -> list[str]:
    return [
        f'<text class="chart-axis-label" x="{x_at(i):.1f}" y="{height - 6}" text-anchor="middle">'
        f"{html_.escape(str(label))}</text>"
        for i, label in enumerate(x_labels)
    ]


def _render_chart_legend(names: list[str]) -> str:
    items = "".join(
        f'<span class="legend-item"><span class="legend-dot" '
        f'style="background:{palette.PALETTE[i % len(palette.PALETTE)]}"></span>'
        f"{html_.escape(name)}</span>"
        for i, name in enumerate(names)
    )
    return f'<div class="chart-legend">{items}</div>'


def render_line_chart(
    chart_id: str,
    series: dict[str, list[float | None]],
    x_labels: list[str],
    y_domain: tuple[float, float] | None = None,
    y_min: float | None = None,
    y_format: Callable[[float], str] | None = None,
    show_legend: bool = True,
) -> str:
    all_values = [v for values in series.values() for v in values if v is not None]
    if not all_values:
        return '<div class="empty">no data</div>'

    if y_domain is not None:
        lo, hi = y_domain
    else:
        lo = y_min if y_min is not None else min(0.0, min(all_values))
        hi = max(all_values)
        hi = hi + (hi - lo) * 0.12 if hi > lo else lo + 1

    y_format = y_format or format_metric_value

    width, height = 640, 220
    pad_left, pad_right, pad_top, pad_bottom = 48, 16, 14, 26
    inner_w = width - pad_left - pad_right
    inner_h = height - pad_top - pad_bottom
    n = len(x_labels)

    def x_at(i: int) -> float:
        return pad_left + (inner_w * i / (n - 1) if n > 1 else inner_w / 2)

    def y_at(v: float) -> float:
        if hi == lo:
            return pad_top + inner_h / 2
        return pad_top + inner_h - ((v - lo) / (hi - lo)) * inner_h

    svg_parts = [
        f'<line class="chart-axis" x1="{pad_left}" y1="{pad_top}" '
        f'x2="{pad_left}" y2="{pad_top + inner_h}" />',
        f'<line class="chart-axis" x1="{pad_left}" y1="{pad_top + inner_h}" '
        f'x2="{pad_left + inner_w}" y2="{pad_top + inner_h}" />',
        f'<text class="chart-axis-label" x="4" y="{pad_top + 4}">'
        f"{html_.escape(y_format(hi))}</text>",
        f'<text class="chart-axis-label" x="4" y="{pad_top + inner_h}">'
        f"{html_.escape(y_format(lo))}</text>",
    ]
    svg_parts.extend(_x_axis_ticks_svg(x_labels, x_at, height))

    series_colors = {
        name: palette.PALETTE[idx % len(palette.PALETTE)] for idx, name in enumerate(series.keys())
    }
    svg_parts.extend(_line_series_svg(series, series_colors, x_labels, x_at, y_at, y_format))

    svg = (
        f'<svg id="{chart_id}" class="chart-svg" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="xMidYMid meet">' + "".join(svg_parts) + "</svg>"
    )

    tooltips = _line_chart_tooltips(x_labels, series, series_colors, y_format)
    hover_data = embed_hover_data(
        chart_id,
        [round(x_at(i), 1) for i in range(n)],
        pad_top,
        pad_top + inner_h,
        tooltips,
    )

    legend = _render_chart_legend(list(series.keys())) if show_legend and len(series) > 1 else ""

    return svg + hover_data + legend


def _line_series_svg(
    series: dict[str, list[float | None]],
    series_colors: dict[str, str],
    x_labels: list[str],
    x_at: Callable[[int], float],
    y_at: Callable[[float], float],
    y_format: Callable[[float], str],
) -> list[str]:
    parts = []
    for name, values in series.items():
        color = series_colors[name]
        points = [(i, v) for i, v in enumerate(values) if v is not None]

        if len(points) >= 2:
            path = " ".join(f"{x_at(i):.1f},{y_at(v):.1f}" for i, v in points)
            parts.append(f'<polyline class="chart-line" points="{path}" style="stroke:{color}" />')

        for i, v in points:
            parts.append(
                f'<circle class="chart-point" cx="{x_at(i):.1f}" cy="{y_at(v):.1f}" r="3.5" '
                f'style="fill:{color}">'
                f"<title>{html_.escape(str(name))}: {html_.escape(y_format(v))} "
                f"(epoch {html_.escape(str(x_labels[i]))})</title>"
                f"</circle>"
            )
    return parts


def _line_chart_tooltips(
    x_labels: list[str],
    series: dict[str, list[float | None]],
    series_colors: dict[str, str],
    y_format: Callable[[float], str],
) -> list[dict[str, Any]]:
    return [
        {
            "title": f"Epoch {x_labels[i]}",
            "rows": [
                {"color": series_colors[name], "label": name, "value": y_format(value)}
                for name, values in series.items()
                if (value := values[i]) is not None
            ],
        }
        for i in range(len(x_labels))
    ]


def render_stacked_area_chart(
    chart_id: str,
    layers: list[tuple[str, list[float | None]]],
    x_labels: list[str],
    y_format: Callable[[float], str] | None = None,
    height: int = 340,
) -> str:
    clean_layers = [
        (name, [v if v is not None else 0.0 for v in values]) for name, values in layers
    ]
    if not any(any(values) for _, values in clean_layers):
        return '<div class="empty">no data</div>'

    n = len(x_labels)
    totals = [sum(values[i] for _, values in clean_layers) for i in range(n)]
    y_format = y_format or format_metric_value

    width = 640
    pad_left, pad_right, pad_top, pad_bottom = 52, 16, 16, 30
    inner_w = width - pad_left - pad_right
    inner_h = height - pad_top - pad_bottom
    hi = (max(totals) or 1) * 1.12

    def x_at(i: int) -> float:
        return pad_left + (inner_w * i / (n - 1) if n > 1 else inner_w / 2)

    def y_at(v: float) -> float:
        return pad_top + inner_h - (v / hi) * inner_h

    svg_parts = [
        f'<line class="chart-axis" x1="{pad_left}" y1="{pad_top}" '
        f'x2="{pad_left}" y2="{pad_top + inner_h}" />',
        f'<line class="chart-axis" x1="{pad_left}" y1="{pad_top + inner_h}" '
        f'x2="{pad_left + inner_w}" y2="{pad_top + inner_h}" />',
        f'<text class="chart-axis-label" x="4" y="{pad_top + 4}">'
        f"{html_.escape(y_format(hi))}</text>",
        f'<text class="chart-axis-label" x="4" y="{pad_top + inner_h}">0</text>',
    ]
    svg_parts.extend(_gridlines_svg(hi, y_at, pad_left, inner_w, y_format))
    svg_parts.extend(_x_axis_ticks_svg(x_labels, x_at, height))

    layer_colors = [palette.PALETTE[idx % len(palette.PALETTE)] for idx in range(len(clean_layers))]
    svg_parts.extend(
        _stacked_area_svg(clean_layers, layer_colors, x_labels, x_at, y_at, totals, y_format)
    )

    svg = (
        f'<svg id="{chart_id}" class="chart-svg" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="xMidYMid meet">' + "".join(svg_parts) + "</svg>"
    )

    tooltips = _stacked_area_tooltips(x_labels, clean_layers, layer_colors, totals, y_format)
    hover_data = embed_hover_data(
        chart_id,
        [round(x_at(i), 1) for i in range(n)],
        pad_top,
        pad_top + inner_h,
        tooltips,
    )

    legend = _render_chart_legend([name for name, _ in clean_layers])

    return svg + hover_data + legend


def _gridlines_svg(
    hi: float,
    y_at: Callable[[float], float],
    pad_left: int,
    inner_w: int,
    y_format: Callable[[float], str],
) -> list[str]:
    step = nice_step(hi)
    parts = []
    gridline_value = step
    while gridline_value < hi:
        gridline_y = y_at(gridline_value)
        parts.append(
            f'<line class="chart-gridline" x1="{pad_left}" y1="{gridline_y:.1f}" '
            f'x2="{pad_left + inner_w}" y2="{gridline_y:.1f}" />'
        )
        parts.append(
            f'<text class="chart-axis-label" x="4" y="{gridline_y + 3:.1f}">'
            f"{html_.escape(y_format(gridline_value))}</text>"
        )
        gridline_value += step
    return parts


def _stacked_area_svg(
    clean_layers: list[tuple[str, list[float]]],
    layer_colors: list[str],
    x_labels: list[str],
    x_at: Callable[[int], float],
    y_at: Callable[[float], float],
    totals: list[float],
    y_format: Callable[[float], str],
) -> list[str]:
    n = len(x_labels)
    parts = []
    cumulative_lower = [0.0] * n
    for idx, (name, values) in enumerate(clean_layers):
        color = layer_colors[idx]
        cumulative_upper = [cumulative_lower[i] + values[i] for i in range(n)]

        top_points = [(x_at(i), y_at(cumulative_upper[i])) for i in range(n)]
        bottom_points = [(x_at(i), y_at(cumulative_lower[i])) for i in reversed(range(n))]
        polygon = " ".join(f"{x:.1f},{y:.1f}" for x, y in top_points + bottom_points)
        parts.append(f'<polygon class="chart-area" points="{polygon}" style="fill:{color}" />')

        if n > 1:
            line = " ".join(f"{x:.1f},{y:.1f}" for x, y in top_points)
            parts.append(f'<polyline class="chart-line" points="{line}" style="stroke:{color}" />')

        for i, (x, y) in enumerate(top_points):
            parts.append(
                f'<circle class="chart-point" cx="{x:.1f}" cy="{y:.1f}" r="3.5" '
                f'style="fill:{color}">'
                f"<title>{html_.escape(name)}: {html_.escape(y_format(values[i]))} "
                f"(epoch {html_.escape(str(x_labels[i]))}) - "
                f"total {html_.escape(y_format(totals[i]))}</title>"
                f"</circle>"
            )

        cumulative_lower = cumulative_upper
    return parts


def _stacked_area_tooltips(
    x_labels: list[str],
    clean_layers: list[tuple[str, list[float]]],
    layer_colors: list[str],
    totals: list[float],
    y_format: Callable[[float], str],
) -> list[dict[str, Any]]:
    return [
        {
            "title": f"Epoch {x_labels[i]}",
            "rows": [
                {"color": layer_colors[idx], "label": name, "value": y_format(values[i])}
                for idx, (name, values) in enumerate(clean_layers)
            ]
            + [{"label": "Total", "value": y_format(totals[i]), "strong": True}],
        }
        for i in range(len(x_labels))
    ]
