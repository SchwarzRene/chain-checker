from chain_checker.baseclasses.corpus.output import ModelOutput


class EmptyModelOutput(ModelOutput):
    """Placeholder prediction assigned to every `Entry` before a chain has
    run on it (see `Corpus.load`/`Corpus.reset`), so callers can call
    `get_model_output()` on a fresh entry without a None-check."""

    def __init__(self) -> None:
        super().__init__("", {})
