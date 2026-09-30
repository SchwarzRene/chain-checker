from chain_checker.utils.report.css import (
    epoch_report_css,
    shared_css,
    summary_report_css,
)


def test_bar_and_histogram_containers_scroll_instead_of_overflowing_their_card():
    css = epoch_report_css()

    bars_rule = css.split(".bars {", 1)[1].split("}", 1)[0]
    assert "overflow-y: auto" in bars_rule

    histogram_rule = css.split(".histogram {", 1)[1].split("}", 1)[0]
    assert "overflow-x: auto" in histogram_rule


def test_run_info_chips_are_styled_via_shared_css_not_just_the_epoch_report():
    # Regression: this used to live only in epoch_report_css(), so the run-info
    # chips summary_report.py renders (modifier info, chain info) had no
    # styling at all on the training summary page - shared_css() is what both
    # report types actually include.
    assert ".run-info-chip" in shared_css()


def test_collapsed_prompt_styling_is_shared_not_just_the_summary_report():
    # Regression: this used to live only in summary_report_css() ("Prompt
    # evolution"), so epoch_report.py's own collapse_prompt=True usage
    # (runs_overview.html) would render an unstyled <details> block.
    assert "details.prompt-entry" in shared_css()


def test_every_stylesheet_is_non_empty_with_balanced_braces():
    for css in (shared_css(), epoch_report_css(), summary_report_css()):
        assert css.strip()
        assert css.count("{") == css.count("}")
