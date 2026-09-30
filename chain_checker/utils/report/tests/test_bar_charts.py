import pytest

from chain_checker.baseclasses.metrics.m_value import ValueMetrics
from chain_checker.utils.report import palette
from chain_checker.utils.report.bar_charts import (
    render_accuracy_bars,
    render_agreement,
    render_bars,
    render_confusion,
    render_histogram,
    render_paired_bars,
)


def _agreement(*pairs) -> dict:
    metric = ValueMetrics("tonality", "d")
    for true_value, pred_value in pairs:
        metric.add(true_value, pred_value)
    return metric.compute()


# ---- render_bars ----


def test_render_bars_shows_an_empty_note_for_no_counts():
    assert "no data" in render_bars({})


def test_render_bars_scales_each_bar_against_the_largest_count():
    html = render_bars({"alice": 4, "bob": 2})

    assert "width:100.0%" in html
    assert "width:50.0%" in html


def test_render_bars_escapes_labels_and_appends_the_suffix():
    html = render_bars({"<b>alice</b>": 3}, suffix="%")

    assert "<b>alice</b>" not in html
    assert "&lt;b&gt;alice&lt;/b&gt;" in html
    assert "3%" in html


# ---- render_accuracy_bars ----


def test_render_accuracy_bars_shows_an_empty_note_for_no_runs():
    assert "no data" in render_accuracy_bars({})


def test_render_accuracy_bars_scales_against_a_fixed_0_to_100_axis_not_the_other_runs():
    # A weak run (12%) next to a strong one (96%) must not stretch the weak
    # bar to fill the track the way render_bars() would - each bar's width
    # is that run's own accuracy against a fixed 100, not against the max.
    html = render_accuracy_bars({"run_0": 12.0, "run_1": 96.0})

    assert "width:12.0%" in html
    assert "width:96.0%" in html


def test_render_accuracy_bars_colors_each_bar_by_its_own_accuracy():
    html = render_accuracy_bars({"run_0": 40.0, "run_1": 65.0, "run_2": 90.0})

    assert f"--bar-color:{palette.FALSE_COLOR}" in html
    assert f"--bar-color:{palette.MID_COLOR}" in html
    assert f"--bar-color:{palette.TRUE_COLOR}" in html


def test_render_accuracy_bars_shows_the_run_name_and_percentage():
    html = render_accuracy_bars({"run_0": 83.4})

    assert "run_0" in html
    assert "83.4%" in html


# ---- render_confusion ----


def test_render_confusion_places_diagonal_cells_in_the_true_color():
    html = render_confusion("brand-uppercase", "80%", tt=4, tf=1, ft=1, ff=4)

    assert f"color:{palette.TRUE_COLOR}" in html
    assert f"color:{palette.FALSE_COLOR}" in html
    assert "80%" in html


def test_render_confusion_escapes_the_rule_name():
    html = render_confusion("<script>", "0%", 0, 0, 0, 0)

    assert "<script>" not in html


# ---- render_histogram ----


def test_render_histogram_shows_an_empty_note_for_no_values():
    assert "no data" in render_histogram([])


def test_render_histogram_bins_by_value_and_scales_to_the_tallest_bin():
    html = render_histogram([12, 12, 12, 45])

    assert ">3<" in html
    assert "height:100.0%" in html
    assert "height:33.3%" in html


def test_render_histogram_sorts_labels_instead_of_insertion_order():
    html = render_histogram([45, 12, 12])

    assert html.index(">12<") < html.index(">45<")


# ---- render_paired_bars ----


def test_render_paired_bars_shows_an_empty_note_for_no_values():
    assert "no data" in render_paired_bars([], [])


def test_render_paired_bars_scales_against_the_largest_of_either_series():
    html = render_paired_bars([1.0, 2.0], [4.0, 1.0])

    assert "height:100.0%" in html
    assert "height:25.0%" in html


def test_render_paired_bars_renders_a_missing_score_as_a_dash_not_zero():
    html = render_paired_bars([None], [3.0])

    assert '<div class="paired-value">-</div>' in html
    assert "height:0%" in html


def test_render_paired_bars_requires_equal_length_series():
    with pytest.raises(ValueError):
        render_paired_bars([1.0, 2.0], [1.0])


# ---- render_agreement ----


def test_render_agreement_shows_both_counts_the_rule_name_and_accuracy():
    html = render_agreement("tonality", _agreement(("formal", "formal"), ("formal", "casual")))

    assert "tonality" in html
    assert "50%" in html
    assert "Matched" in html and "Mismatched" in html


def test_render_agreement_scales_the_two_bars_against_each_other():
    html = render_agreement("tonality", _agreement(("a", "a"), ("a", "a"), ("a", "a"), ("a", "b")))

    assert "width:75.0%" in html
    assert "width:25.0%" in html


def test_render_agreement_fills_one_bar_and_empties_the_other_on_a_perfect_rule():
    html = render_agreement("tonality", _agreement(("a", "a")))

    assert "width:100.0%" in html
    assert "width:0.0%" in html
    assert "100%" in html


def test_render_agreement_shows_an_empty_note_rather_than_dividing_by_zero():
    html = render_agreement("tonality", _agreement())

    assert "no data" in html
    assert "width:" not in html


def test_render_agreement_escapes_the_rule_name():
    html = render_agreement("<script>x</script>", _agreement(("a", "a")))

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
