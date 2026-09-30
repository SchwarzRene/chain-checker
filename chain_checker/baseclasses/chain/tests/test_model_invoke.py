import asyncio

import pytest

from chain_checker.baseclasses.chain.calls import Call
from chain_checker.baseclasses.chain.model import ChainRunError, Model
from chain_checker.utils.console.wait_bar import WaitBar


class _Runnable:
    def __init__(self, *, async_raises=None) -> None:
        self.calls: list[str] = []
        self.configs: list = []
        self._async_raises = async_raises

    async def ainvoke(self, payload, config=None):
        self.calls.append("ainvoke")
        self.configs.append(config)
        if self._async_raises is not None:
            raise self._async_raises
        return {"via": "ainvoke", "payload": payload}

    def invoke(self, payload, config=None):
        self.calls.append("invoke")
        self.configs.append(config)
        return {"via": "invoke", "payload": payload}


class _SyncOnly:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.configs: list = []

    def invoke(self, payload, config=None):
        self.calls.append("invoke")
        self.configs.append(config)
        return {"via": "invoke", "payload": payload}


class _Neither:
    pass


def _model(runnable) -> Model:
    model = Model.__new__(Model)
    model.runnable = runnable
    model._chain_type = "template_checklist"
    model._wait_bar = WaitBar()
    model._wait_prefix = None
    model._event_loop = None
    return model


def test_a_runnable_with_a_working_ainvoke_is_never_called_synchronously():
    runnable = _Runnable()

    out, cap = _model(runnable)._run_chain({"text": "hi"})

    assert out["via"] == "ainvoke"
    assert runnable.calls == ["ainvoke"]
    assert cap.calls == []


def test_a_runnable_with_no_ainvoke_at_all_goes_straight_to_invoke():
    runnable = _SyncOnly()

    out, _ = _model(runnable)._run_chain({"text": "hi"})

    assert out["via"] == "invoke"
    assert runnable.calls == ["invoke"]


@pytest.mark.parametrize("runnable_type", [_Runnable, _SyncOnly], ids=["async", "sync"])
def test_the_payload_reaches_whichever_path_runs(runnable_type):
    out, _ = _model(runnable_type())._run_chain({"text": "hi"})

    assert out["payload"] == {"text": "hi"}


@pytest.mark.parametrize("runnable_type", [_Runnable, _SyncOnly], ids=["async", "sync"])
def test_the_analyser_is_wired_in_as_a_callback_on_whichever_path_runs(runnable_type):

    runnable = runnable_type()

    _, cap = _model(runnable)._run_chain({"text": "hi"})

    assert runnable.configs == [{"callbacks": [cap]}]


@pytest.mark.parametrize(
    "error",
    [
        ValueError("could not parse into OutputSchema"),
        AttributeError("module 'services' has no attribute 'store_file'"),
        RuntimeError("the chain itself broke"),
        KeyError("text"),
        NotImplementedError("user code raised before any LLM call"),
    ],
    ids=["parse-failure", "missing-service", "runtime", "key-error", "not-implemented"],
)
def test_an_error_from_inside_the_chain_is_not_retried_synchronously(error):

    runnable = _Runnable(async_raises=error)

    with pytest.raises(RuntimeError, match="raised while running"):
        _model(runnable)._run_chain({"text": "hi"})

    assert runnable.calls == ["ainvoke"]


def test_the_original_error_is_what_gets_reported():
    runnable = _Runnable(async_raises=ValueError("the real cause"))

    with pytest.raises(RuntimeError, match="the real cause") as excinfo:
        _model(runnable)._run_chain({"text": "hi"})

    assert isinstance(excinfo.value.__cause__, ValueError)


def test_calls_the_analyser_already_recorded_survive_on_the_raised_error():
    # Simulates a structured-output JSON parsing failure: the underlying
    # chat model call succeeds (on_llm_end fires and the callback records
    # it) and only the downstream parser then rejects the reply - the raw
    # prompt/reply must not be lost along with the exception.
    class _RunnableRecordingThenFailing:
        async def ainvoke(self, payload, config=None):
            cap = config["callbacks"][0]
            cap.calls.append(
                Call(prompt=[], reply="not valid json", model_name="qwen", token_usage={})
            )
            raise ValueError("Invalid json output: not valid json")

    with pytest.raises(ChainRunError) as excinfo:
        _model(_RunnableRecordingThenFailing())._run_chain({"text": "hi"})

    assert [call.reply for call in excinfo.value.calls] == ["not valid json"]


def test_no_calls_recorded_means_an_empty_calls_list_on_the_raised_error():
    runnable = _Runnable(async_raises=RuntimeError("never reached the model"))

    with pytest.raises(ChainRunError) as excinfo:
        _model(runnable)._run_chain({"text": "hi"})

    assert excinfo.value.calls == []


def test_a_runnable_with_neither_method_is_named_in_the_error():
    with pytest.raises(RuntimeError, match="neither ainvoke\\(\\) nor invoke\\(\\)"):
        _model(_Neither())._run_chain({"text": "hi"})


def test_an_ainvoke_capable_chain_called_from_inside_a_loop_raises_clearly():
    runnable = _Runnable()
    model = _model(runnable)

    async def _run():
        return model._run_chain({"text": "hi"})

    with pytest.raises(RuntimeError, match="must be called from sync code"):
        asyncio.run(_run())

    assert runnable.calls == []


def test_a_sync_only_chain_still_runs_through_invoke_from_inside_a_loop():
    runnable = _SyncOnly()
    model = _model(runnable)

    async def _run():
        return model._run_chain({"text": "hi"})

    out, _ = asyncio.run(_run())

    assert out["via"] == "invoke"
    assert runnable.calls == ["invoke"]
