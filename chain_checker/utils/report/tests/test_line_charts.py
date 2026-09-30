import json

from chain_checker.utils.report import palette
from chain_checker.utils.report.line_charts import (
    embed_hover_data,
    render_line_chart,
    render_stacked_area_chart,
)

# ---- embed_hover_data ----


def test_embed_hover_data_carries_the_payload_as_json():
    html = embed_hover_data("chart-1", [10.0, 20.0], 5.0, 100.0, [{"title": "Epoch 0"}])

    assert 'data-chart-points="chart-1"' in html
    payload = json.loads(html.split(">", 1)[1].rsplit("<", 1)[0])
    assert payload == {
        "xs": [10.0, 20.0],
        "top": 5.0,
        "bottom": 100.0,
        "tooltips": [{"title": "Epoch 0"}],
    }


def test_embed_hover_data_escapes_a_closing_script_tag_in_the_payload():
    html = embed_hover_data("chart-1", [], 0.0, 0.0, [{"title": "</script><script>evil"}])

    assert "</script><script>evil" not in html
    assert "<\\/script><script>evil" in html


# ---- render_line_chart ----


def test_render_line_chart_shows_an_empty_note_when_every_series_is_empty():
    assert "no data" in render_line_chart("chart-1", {"Accuracy": [None, None]}, ["0", "1"])


def test_render_line_chart_draws_one_point_per_non_none_value():
    html = render_line_chart("chart-1", {"Accuracy": [50.0, None, 75.0]}, ["0", "1", "2"])

    assert html.count('<circle class="chart-point"') == 2


def test_render_line_chart_only_draws_a_line_with_at_least_two_points():
    single_point = render_line_chart("chart-1", {"Accuracy": [50.0, None]}, ["0", "1"])
    two_points = render_line_chart("chart-1", {"Accuracy": [50.0, 60.0]}, ["0", "1"])

    assert "<polyline" not in single_point
    assert "<polyline" in two_points


def test_render_line_chart_respects_an_explicit_y_domain_over_the_data_range():
    with_domain = render_line_chart("chart-1", {"Accuracy": [50.0]}, ["0"], y_domain=(0, 100))
    without_domain = render_line_chart("chart-1", {"Accuracy": [50.0]}, ["0"])

    assert ">100</text>" in with_domain
    assert ">100</text>" not in without_domain


def test_render_line_chart_uses_y_min_as_the_lower_bound_when_given():
    html = render_line_chart("chart-1", {"Accuracy": [50.0]}, ["0"], y_min=-10)

    assert ">-10.00</text>" in html


def test_render_line_chart_only_shows_a_legend_with_more_than_one_series():
    one_series = render_line_chart("chart-1", {"Accuracy": [1.0, 2.0]}, ["0", "1"])
    two_series = render_line_chart(
        "chart-1", {"Accuracy": [1.0, 2.0], "MAE": [0.1, 0.2]}, ["0", "1"]
    )

    assert "chart-legend" not in one_series
    assert "chart-legend" in two_series


def test_render_line_chart_can_suppress_the_legend_even_with_multiple_series():
    html = render_line_chart(
        "chart-1",
        {"Accuracy": [1.0, 2.0], "MAE": [0.1, 0.2]},
        ["0", "1"],
        show_legend=False,
    )

    assert "chart-legend" not in html


def test_render_line_chart_assigns_palette_colors_by_series_order():
    html = render_line_chart("chart-1", {"A": [1.0, 2.0], "B": [3.0, 4.0]}, ["0", "1"])

    assert f"stroke:{palette.PALETTE[0]}" in html
    assert f"stroke:{palette.PALETTE[1]}" in html


# ---- render_stacked_area_chart ----


def test_render_stacked_area_chart_shows_an_empty_note_when_every_layer_is_zero():
    html = render_stacked_area_chart(
        "chart-1", [("Prompt", [0, 0]), ("Completion", [0, 0])], ["0", "1"]
    )

    assert "no data" in html


def test_render_stacked_area_chart_draws_one_polygon_per_layer():
    html = render_stacked_area_chart(
        "chart-1", [("Prompt", [100, 120]), ("Completion", [50, 60])], ["0", "1"]
    )

    assert html.count("<polygon") == 2


def test_render_stacked_area_chart_lists_every_layer_in_the_legend():
    html = render_stacked_area_chart("chart-1", [("Prompt", [100]), ("Completion", [50])], ["0"])

    assert "Prompt" in html
    assert "Completion" in html


def test_render_stacked_area_chart_treats_a_missing_value_as_zero_not_a_gap():
    html = render_stacked_area_chart("chart-1", [("Prompt", [100, None])], ["0", "1"])

    assert "no data" not in html
