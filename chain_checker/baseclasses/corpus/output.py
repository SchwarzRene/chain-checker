from typing import Any

from chain_checker.baseclasses.corpus.label import Label


class ModelOutput(Label):
    """A chain's prediction for one entry. Extends `Label`, not `Record`
    directly, so a prediction can be compared with the same
    `matches()`/`get_value()` API used for the expected output, while also
    carrying the two things only a prediction has: the raw LLM conversation
    transcript and the token usage it cost to produce."""

    def __init__(
        self,
        conversation: str,
        output: dict[str, Any],
        token_usage: dict[str, Any] | None = None,
    ) -> None:
        self._conversation = conversation

        self._token_usage: dict[str, Any] = dict(token_usage or {})

        super().__init__(output)

    def set_convo(self, conversation: str) -> None:
        self._conversation = conversation

    def set_output(self, output: dict[str, Any]) -> None:
        self.set(output)

    def get_convo(self) -> str:
        return self._conversation

    def get_output(self) -> dict[str, Any]:
        return self.get()

    def get_token_usage(self) -> dict[str, Any]:
        return dict(self._token_usage)
