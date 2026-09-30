import pytest

from chain_checker.modifier.llm.llm_baseclass import LLM


class FakeLLM(LLM):
    """A modifier backend that returns a scripted reply and a fixed usage
    figure instead of calling a real model, plus records every prompt it was
    asked to rewrite so a test can assert on what history it saw."""

    def __init__(self, reply: str = "rewritten prompt", usage: tuple[int, int] = (10, 5)) -> None:
        super().__init__()
        self._reply = reply
        self._usage = usage
        self.seen_prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.seen_prompts.append(prompt)
        self._record_usage(*self._usage)
        return self._reply

    def get_run_info(self) -> dict[str, str]:
        return {"backend": "fake"}


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def make_metric_data():
    """Builds one epoch's `data` list in the shape ModifierModel.replay_run
    expects: one accuracy metric plus the two metrics the modifier always
    looks for by name."""

    def _make(
        *,
        accuracy: float = 1.0,
        mispredicted: dict | None = None,
        failed_ids: list | None = None,
        excluded_metric: str | None = None,
    ) -> list[dict]:
        data = [
            {
                "name": "Accuracy-Metrics",
                "description": "share of cases the chain got right",
                "results": {"accuracy": accuracy},
            },
            {
                "name": "Negative-Predicted-Metrics",
                "results": mispredicted or {},
            },
            {
                "name": "Parsing-Metrics",
                "results": {"failed_ids": failed_ids or []},
            },
        ]
        if excluded_metric:
            data.append({"name": excluded_metric, "results": {"whatever": True}})
        return data

    return _make
