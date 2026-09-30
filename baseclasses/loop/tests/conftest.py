import pytest

from chain_checker.baseclasses.corpus.corpus import Corpus
from chain_checker.baseclasses.corpus.entry import Entry
from chain_checker.baseclasses.corpus.input import Input
from chain_checker.baseclasses.corpus.label import Label
from chain_checker.baseclasses.corpus.output import ModelOutput
from chain_checker.baseclasses.corpus.output_empty import EmptyModelOutput

NO_PROMPT = object()


class FakeModel:
    def __init__(
        self,
        *,
        prompt="the system prompt",
        tier="fast",
        output=None,
        answers=None,
        token_usage=None,
        fails=(),
        prompt_error=None,
    ) -> None:
        self.inputs: list = []
        self.prompt = prompt
        self._tier = tier
        self._output = {"passed": True} if output is None else output

        self._answers = dict(answers or {})
        self._token_usage = {"total_tokens": 5} if token_usage is None else token_usage
        self._fails = set(fails)
        self._prompt_error = prompt_error

    def get_type(self) -> str:
        return "tonality"

    def get_chain_type(self) -> str:
        return "template_checklist"

    def get_config(self) -> dict:
        return {"type": "tonality", "chain": "template_checklist", "tier": self._tier}

    def set_tier(self, tier: str) -> None:
        self._tier = tier

    def set_fails(self, *entry_ids: str) -> None:
        self._fails = set(entry_ids)

    def get_system_prompt(self) -> str:
        if self._prompt_error is not None:
            raise self._prompt_error
        if self.prompt is NO_PROMPT:
            raise AttributeError("has no module-level SYSTEM_PROMPT constant")
        return self.prompt

    def __call__(self, inp) -> ModelOutput:
        entry_id = inp.get_value("text")
        self.inputs.append(entry_id)
        if entry_id in self._fails:
            raise RuntimeError(f"could not parse the reply for entry {entry_id!r}")
        return ModelOutput(
            f"the conversation for {entry_id}",
            dict(self._answers.get(entry_id, self._output)),
            dict(self._token_usage),
        )

    def call_with_progress(self, inp, prefix: str, tracker):
        try:
            result = self(inp)
        except Exception as e:
            tracker.record(0.0)
            print(f"{prefix} - failed: {e}")
            return None, e, 0.0
        tracker.record(0.0)
        print(f"{prefix} - {tracker.get_summary(0.0)}")
        return result, None, 0.0


@pytest.fixture
def model():
    return FakeModel()


@pytest.fixture
def make_corpus():
    def _make(*entry_ids) -> Corpus:
        corpus = Corpus()
        for entry_id in entry_ids:
            corpus.add_entry(
                Entry(
                    entry_id,
                    Input({"text": str(entry_id)}),
                    Label({"passed": True}),
                    {},
                    EmptyModelOutput(),
                )
            )
        return corpus

    return _make


@pytest.fixture
def base_dir(tmp_path) -> str:

    return str(tmp_path / "temp")
