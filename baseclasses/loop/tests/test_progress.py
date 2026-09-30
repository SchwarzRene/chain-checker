import pytest

from chain_checker.baseclasses.loop.progress import (
    ProgressTracker,
    format_duration,
)


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "0s"),
        (0.4, "0s"),
        (1.6, "2s"),
        (59, "59s"),
        (60, "1m 0s"),
        (95, "1m 35s"),
        (3599, "59m 59s"),
        (3600, "1h 0m"),
        (7860, "2h 11m"),
    ],
    ids=[
        "zero",
        "sub-second",
        "rounded-up",
        "just-under-a-minute",
        "a-minute",
        "minutes-and-seconds",
        "just-under-an-hour",
        "an-hour",
        "hours",
    ],
)
def test_a_duration_reads_in_the_largest_unit_it_needs(seconds, expected):
    assert format_duration(seconds) == expected


def test_a_negative_duration_reads_as_zero():

    assert format_duration(-5) == "0s"


def test_a_fresh_tracker_has_processed_nothing():
    tracker = ProgressTracker(10)

    assert tracker.get_done() == 0
    assert tracker.get_remaining() == 10


def test_an_average_over_no_entries_is_zero_rather_than_a_crash():

    assert ProgressTracker(10).get_average() == 0.0


def test_each_recorded_entry_counts_towards_done_and_off_remaining():
    tracker = ProgressTracker(3)

    tracker.record(1.0)
    tracker.record(2.0)

    assert tracker.get_done() == 2
    assert tracker.get_remaining() == 1


def test_the_average_is_over_every_entry_recorded():
    tracker = ProgressTracker(3)

    tracker.record(1.0)
    tracker.record(3.0)

    assert tracker.get_average() == 2.0


def test_a_run_that_processed_more_than_it_announced_has_nothing_remaining():

    tracker = ProgressTracker(1)

    tracker.record(1.0)
    tracker.record(1.0)

    assert tracker.get_remaining() == 0


def test_the_line_reports_the_entry_the_progress_and_the_estimate():
    tracker = ProgressTracker(4)
    tracker.record(60.0)

    line = tracker.get_line("case-1", 60.0)

    assert "(1/4 processed)" in line
    assert "entry 'case-1'" in line
    assert "done in 1m 0s" in line
    assert "avg 1m 0s/entry" in line
    assert "3 left" in line
    assert "~3m 0s remaining" in line


def test_the_estimate_is_the_average_across_what_is_left():
    tracker = ProgressTracker(11)
    tracker.record(10.0)
    tracker.record(20.0)

    assert "~2m 15s remaining" in tracker.get_line("case-2", 20.0)


def test_the_last_entry_of_a_run_estimates_nothing_left():
    tracker = ProgressTracker(1)
    tracker.record(30.0)

    line = tracker.get_line("case-1", 30.0)

    assert "0 left" in line
    assert "~0s remaining" in line
