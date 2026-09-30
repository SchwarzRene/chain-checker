import types

import pytest

from chain_checker.baseclasses.chain.calls import Call
from chain_checker.baseclasses.chain.model import ChainRunError, Model, _dump
from chain_checker.baseclasses.corpus.input import Input
from chain_checker.baseclasses.loop.progress import ProgressTracker
from chain_checker.utils.console.wait_bar import WaitBar


class _Message:
    def __init__(self, type, content) -> None:
        self.type = type
        self.content = content


class _Payload:
    def __init__(self, data) -> None:
        self._data = data

    def model_dump(self):
        return self._data


class _InputSchema:
    def __init__(self, **data) -> None:
        self._data = data

    def model_dump(self):
        return dict(self._data)


class _Chain:
    InputSchema = _InputSchema


class _Runnable:
    def __init__(self, out, calls=()) -> None:
        self.payloads: list = []
        self._out = out
        self._calls = list(calls)

    async def ainvoke(self, payload, config=None):

        self.payloads.append(payload)
        cap = config["callbacks"][0]
        cap.calls.extend(self._calls)
        return self._out


def _call(usage=None, model_name="gpt-4o-mini", reply="the raw reply") -> Call:
    return Call([_Message("system", "be brief")], reply, model_name, usage or {})


def _model(runnable=None) -> Model:
    model = Model.__new__(Model)
    model.chain = _Chain()
    model.runnable = runnable
    model._chain_type = "template_checklist"
    model._last_model_names = []
    model._last_token_usage = {}
    model._wait_bar = WaitBar()
    model._wait_prefix = None
    model._event_loop = None
    return model


def test_a_pydantic_chain_output_becomes_its_dict():
    assert _dump(_Payload({"verdict": "pass"})) == {"verdict": "pass"}


def test_a_langgraph_dict_is_walked_for_models_nested_inside_it():

    out = _dump({"checked": _Payload({"ok": True}), "note": "kept"})

    assert out == {"checked": {"ok": True}, "note": "kept"}


def test_models_inside_a_list_are_walked_too():
    assert _dump([_Payload({"i": 1}), _Payload({"i": 2})]) == [{"i": 1}, {"i": 2}]


def test_nesting_is_followed_all_the_way_down():
    value = {"groups": [{"items": [_Payload({"i": 1})]}]}

    assert _dump(value) == {"groups": [{"items": [{"i": 1}]}]}


@pytest.mark.parametrize("value", ["text", 1, 1.5, True, None, [], {}], ids=lambda v: repr(v))
def test_plain_data_passes_through_unchanged(value):
    assert _dump(value) == value


def test_the_inputs_fields_reach_the_chain_as_its_schema_dumped_them():
    runnable = _Runnable({"verdict": "pass"})

    _model(runnable)(Input({"text": "hi", "language": "en"}))

    assert runnable.payloads == [{"text": "hi", "language": "en"}]


@pytest.mark.parametrize("wrong", [{"text": "hi"}, "hi", None], ids=["raw-dict", "str", "none"])
def test_being_handed_something_other_than_an_input_ends_the_run(wrong):

    with pytest.raises(SystemExit):
        _model(_Runnable({}))(wrong)


def test_an_input_the_schema_refuses_names_the_chain_and_the_input():
    class _Strict:
        class InputSchema:
            def __init__(self, **data) -> None:
                raise TypeError("unexpected keyword argument 'nope'")

    model = _model(_Runnable({}))
    model.chain = _Strict()

    with pytest.raises(ValueError, match="does not match chain"):
        model(Input({"nope": 1}))


def test_a_call_returns_the_conversation_the_predictions_and_the_cost():
    usage = {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}
    model = _model(_Runnable(_Payload({"verdict": "pass"}), [_call(usage=usage)]))

    out = model(Input({"text": "hi"}))

    assert out.get_output() == {"verdict": "pass"}
    assert "--- raw reply ---\nthe raw reply" in out.get_convo()
    assert out.get_token_usage() == usage


def test_the_cost_of_the_last_call_is_kept_for_a_run_report():
    usage = {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}
    model = _model(_Runnable({}, [_call(usage=usage), _call(usage=usage)]))

    model(Input({"text": "hi"}))

    assert model.get_last_token_usage()["total_tokens"] == 10


def test_the_models_that_answered_are_kept_for_a_run_report():
    model = _model(_Runnable({}, [_call(model_name="gpt-4o"), _call(model_name="o3")]))

    model(Input({"text": "hi"}))

    assert model._last_model_names == ["gpt-4o", "o3"]


def test_a_second_call_replaces_the_first_calls_cost():

    usage = {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}
    model = _model(_Runnable({}, [_call(usage=usage)]))
    model(Input({"text": "hi"}))

    model.runnable = _Runnable({})
    model(Input({"text": "hi again"}))

    assert model.get_last_token_usage() == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def test_the_reported_usage_cannot_be_rewritten_through_the_getter():
    usage = {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
    model = _model(_Runnable({}, [_call(usage=usage)]))
    model(Input({"text": "hi"}))

    model.get_last_token_usage()["total_tokens"] = 99999

    assert model.get_last_token_usage()["total_tokens"] == 4


def test_an_entry_whose_chain_called_no_model_still_produces_an_output():

    out = _model(_Runnable({"verdict": "pass"}))(Input({"text": "hi"}))

    assert out.get_convo() == ""
    assert out.get_token_usage() == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


@pytest.mark.parametrize(
    "out",
    [["a", "b"], "just text", 3, None, _Payload(["not", "a", "mapping"])],
    ids=["list", "str", "int", "none", "pydantic-list"],
)
def test_a_chain_that_returned_no_named_fields_is_named_in_the_error(out):

    with pytest.raises(ValueError, match="template_checklist"):
        _model(_Runnable(out))(Input({"text": "hi"}))


# --------------------------------------------------------------------------
# call_with_progress: the wait-bar-aware wrapper checker.py and
# ClassificationLoop drive their per-case progress bar through.
# --------------------------------------------------------------------------


class _FailingRunnable:
    async def ainvoke(self, payload, config=None):
        raise RuntimeError("boom")


def test_call_with_progress_returns_the_prediction_and_no_error_on_success():
    model = _model(_Runnable({"verdict": "pass"}))
    tracker = ProgressTracker(1)

    result, error, elapsed = model.call_with_progress(Input({"text": "hi"}), "prefix", tracker)

    assert error is None
    assert result.get_output() == {"verdict": "pass"}
    assert elapsed >= 0
    assert tracker.get_done() == 1


def test_call_with_progress_returns_the_chain_run_error_instead_of_raising():
    model = _model(_FailingRunnable())
    tracker = ProgressTracker(1)

    result, error, elapsed = model.call_with_progress(Input({"text": "hi"}), "prefix", tracker)

    assert result is None
    assert isinstance(error, ChainRunError)
    assert "boom" in str(error)
    assert elapsed >= 0


def test_call_with_progress_records_elapsed_time_even_on_failure():
    model = _model(_FailingRunnable())
    tracker = ProgressTracker(1)

    model.call_with_progress(Input({"text": "hi"}), "prefix", tracker)

    assert tracker.get_done() == 1


def test_call_with_progress_prints_the_bar_with_the_given_prefix(capsys):
    model = _model(_Runnable({"verdict": "pass"}))
    tracker = ProgressTracker(1)

    model.call_with_progress(Input({"text": "hi"}), "my-prefix-marker", tracker)

    assert "my-prefix-marker" in capsys.readouterr().out


def test_call_with_progress_folds_the_trackers_summary_into_the_finish_line(capsys):
    model = _model(_Runnable({"verdict": "pass"}))
    tracker = ProgressTracker(2)

    model.call_with_progress(Input({"text": "hi"}), "prefix", tracker)

    printed = capsys.readouterr().out
    assert "done in" in printed
    assert "1 left" in printed


def test_call_with_progress_clears_the_wait_prefix_afterward():
    model = _model(_Runnable({"verdict": "pass"}))
    tracker = ProgressTracker(1)

    model.call_with_progress(Input({"text": "hi"}), "prefix", tracker)

    assert model._wait_prefix is None


def test_call_with_progress_lets_a_fatal_error_through_uncaught():
    # A BaseException (e.g. CheckerError from fail()) signals a genuine infra
    # problem, not a per-case failure - it must not be swallowed like one.
    class _Fatal(BaseException):
        pass

    class _RunnableRaisingFatal:
        async def ainvoke(self, payload, config=None):
            raise _Fatal("infra problem")

    model = _model(_RunnableRaisingFatal())
    tracker = ProgressTracker(1)

    with pytest.raises(_Fatal):
        model.call_with_progress(Input({"text": "hi"}), "prefix", tracker)


# --------------------------------------------------------------------------
# get_tier: feeds the (cosmetic) tier suffix on check_N/run_N directories.
# --------------------------------------------------------------------------


def test_get_tier_returns_the_chains_own_tier_when_no_override_is_given():
    model = Model.__new__(Model)
    model._tier_override = None
    model.chain = types.SimpleNamespace(tier="balanced")

    assert model.get_tier() == "balanced"


def test_get_tier_prefers_an_explicit_override_over_the_chains_own_tier():
    model = Model.__new__(Model)
    model._tier_override = "fast"
    model.chain = types.SimpleNamespace(tier="balanced")

    assert model.get_tier() == "fast"
