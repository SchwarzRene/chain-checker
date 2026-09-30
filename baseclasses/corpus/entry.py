from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chain_checker.baseclasses.corpus.input import Input
    from chain_checker.baseclasses.corpus.label import Label
    from chain_checker.baseclasses.corpus.output import ModelOutput


class Entry:
    """Pairs one corpus case's input and expected label with the model's
    most recent prediction. The prediction is mutable (`set_model_output`)
    rather than fixed at construction, since the same corpus is re-run
    against a new prediction every epoch without being reloaded."""

    def __init__(
        self,
        entry_id: str | int,
        input_data: Input,
        output: Label,
        info: dict[str, Any],
        predicted: ModelOutput,
    ) -> None:
        self._id: str | int = entry_id
        self._input: Input = input_data
        self._output: Label = output
        self._predicted: ModelOutput = predicted
        self._info: dict[str, Any] = dict(info)

    def __str__(self) -> str:
        return f"Entry {self._id}\n{self._input}\n{self._output}\n{self._predicted}"

    def get_info(self, key: str) -> Any:
        return self._info.get(key)

    def get_id(self) -> str | int:
        return self._id

    def get_input(self) -> Input:
        return self._input

    def get_output(self) -> Label:
        return self._output

    def get_model_output(self) -> ModelOutput:
        return self._predicted

    def set_model_output(self, output: ModelOutput) -> None:
        self._predicted = output
