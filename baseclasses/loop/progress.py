def format_duration(seconds: float) -> str:

    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


class ProgressTracker:
    """Times each entry as it's processed to print a running "done so far /
    average / remaining" line. `total` must count only entries that will
    actually be passed to `record()` - a caller that recalls some entries
    from cache should pass the count excluding those, not the full corpus
    size."""

    def __init__(self, total: int) -> None:
        self._total = total
        self._durations: list[float] = []

    def record(self, seconds: float) -> None:

        self._durations.append(seconds)

    def get_done(self) -> int:
        return len(self._durations)

    def get_remaining(self) -> int:

        return max(0, self._total - len(self._durations))

    def get_average(self) -> float:

        if not self._durations:
            return 0.0
        return sum(self._durations) / len(self._durations)

    def get_summary(self, elapsed: float) -> str:
        average = self.get_average()
        remaining = self.get_remaining()
        return (
            f"done in {format_duration(elapsed)} "
            f"(avg {format_duration(average)}/entry, {remaining} left, "
            f"~{format_duration(average * remaining)} remaining)"
        )

    def get_line(self, entry_id: str, elapsed: float) -> str:
        return (
            f"(LOOP) ({self.get_done()}/{self._total} processed) entry "
            f"'{entry_id}' - {self.get_summary(elapsed)}"
        )
