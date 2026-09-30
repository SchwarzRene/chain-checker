import json
import os

import pytest

from bin.checker import (
    _print_output_shape_suggestion,
    _resolve_check_dir,
    _save_failed_case,
    _score_recalled,
    run_case,
)
from chain_checker.baseclasses.chain.calls import Call
from chain_checker.baseclasses.chain.model import ChainRunError
from chain_checker.baseclasses.corpus.corpus import Corpus
from chain_checker.baseclasses.corpus.entry import Entry
from chain_checker.baseclasses.corpus.input import Input
from chain_checker.baseclasses.corpus.label import Label
from chain_checker.baseclasses.corpus.output import ModelOutput
from chain_checker.baseclasses.corpus.output_empty import EmptyModelOutput
from chain_checker.baseclasses.loop.cache import LoopCache
from chain_checker.baseclasses.loop.progress import ProgressTracker
from chain_checker.utils.errors import CheckerError


def _entry(entry_id="a", expected=None) -> Entry:
    return Entry(
        entry_id,
        Input({"text": str(entry_id)}),
        Label(expected if expected is not None else {"passed": True}),
        {},
        EmptyModelOutput(),
    )


class _FakeModel:
    """Stands in for `Model`: mirrors `call_with_progress`'s own contract
    (`except Exception`, never `BaseException`), since that is what
    `run_case` drives a case through instead of calling `__call__` directly."""

    def __init__(self, *, error: Exception | None = None, output: dict | None = None) -> None:
        self._error = error
        self._output = {"passed": True} if output is None else output

    def __call__(self, _inp) -> ModelOutput:
        if self._error is not None:
            raise self._error
        return ModelOutput("the conversation", dict(self._output), {"total_tokens": 5})

    def call_with_progress(
        self, inp, prefix: str, tracker: ProgressTracker
    ) -> tuple[ModelOutput | None, Exception | None, float]:
        # Prints `prefix` on the way out, same as the real WaitBar.finish()
        # it stands in for, so a test can check what run_case built it from.
        try:
            result = self(inp)
        except Exception as e:
            tracker.record(0.0)
            print(f"{prefix} - failed")
            return None, e, 0.0
        tracker.record(0.0)
        print(f"{prefix} - {tracker.get_summary(0.0)}")
        return result, None, 0.0

    def get_chain_type(self) -> str:
        return "fake-chain"


def _run(model: _FakeModel, entry) -> tuple[bool, Exception | None, float]:
    return run_case(model, entry, 1, 1, ProgressTracker(1))


# --------------------------------------------------------------------------
# run_case: a raising model must not abort the corpus - same resilience as
# trainingLoop's ClassificationLoop._predict.
# --------------------------------------------------------------------------


def test_a_failing_case_does_not_raise():
    ok, error, _ = _run(_FakeModel(error=RuntimeError("bad reply")), _entry())

    assert ok is False
    assert isinstance(error, RuntimeError)


def test_a_failing_case_leaves_the_entry_as_an_empty_prediction():
    entry = _entry()

    _run(_FakeModel(error=RuntimeError("bad reply")), entry)

    assert entry.get_model_output().get_output() == {}


def test_a_failing_case_warns_with_the_case_id_and_reason(capsys):
    _run(_FakeModel(error=RuntimeError("bad reply")), _entry("case-7"))

    printed = capsys.readouterr().out
    assert "case-7" in printed
    assert "bad reply" in printed
    assert "continuing with the rest of the corpus" in printed


def test_a_chain_run_error_leaves_the_prediction_empty_but_keeps_the_conversation():
    # The whole point of ChainRunError: a structured-output parsing failure
    # still has a real prompt/reply to show, even though there is nothing
    # usable to score - unlike a plain exception, which leaves no trace of
    # what was actually sent to or said by the model.
    entry = _entry()
    calls = [Call(prompt=[], reply="not valid json", model_name="qwen", token_usage={})]
    error = ChainRunError("Invalid json output: not valid json", calls)

    _run(_FakeModel(error=error), entry)

    output = entry.get_model_output()
    assert output.get_output() == {}
    assert "not valid json" in output.get_convo()


def test_a_chain_run_error_with_no_calls_leaves_the_entry_as_an_empty_prediction():
    entry = _entry()
    error = ChainRunError("failed before any LLM call", [])

    _run(_FakeModel(error=error), entry)

    assert entry.get_model_output().get_convo() == ""


def test_a_cleaned_reply_on_the_wrapped_cause_gets_its_own_saved_section():
    # Duck-typed: any chain's own parsing exception can carry a
    # `cleaned_reply` attribute (see example_tonality's ChecklistParseError)
    # to get its post-stripping text saved right next to the raw reply,
    # instead of that only ever being visible in the console warning.
    entry = _entry()
    calls = [Call(prompt=[], reply="<think>..</think>{}", model_name="qwen", token_usage={})]
    error = ChainRunError("Chain 'x' raised while running: bad json", calls)
    cause = ValueError("bad json")
    cause.cleaned_reply = "{}"
    error.__cause__ = cause

    _run(_FakeModel(error=error), entry)

    convo = entry.get_model_output().get_convo()
    assert "--- after stripping ---\n{}" in convo


def test_no_cleaned_reply_attribute_means_no_extra_section():
    entry = _entry()
    calls = [Call(prompt=[], reply="not valid json", model_name="qwen", token_usage={})]
    error = ChainRunError("Invalid json output: not valid json", calls)
    error.__cause__ = ValueError("not valid json")

    _run(_FakeModel(error=error), entry)

    assert "after stripping" not in entry.get_model_output().get_convo()


def test_a_fatal_error_is_not_swallowed_as_a_case_failure():
    # fail() raises CheckerError(SystemExit), not Exception - a genuine infra
    # problem (bad app config, missing API key) must still abort the run,
    # the same way ClassificationLoop lets it through untouched.
    with pytest.raises(CheckerError):
        _run(_FakeModel(error=CheckerError("no LiteLLM key configured")), _entry())


# --------------------------------------------------------------------------
# run_case: the ordinary pass/fail path is unaffected by the new resilience.
# --------------------------------------------------------------------------


def test_a_matching_case_passes_with_no_error():
    ok, error, _ = _run(_FakeModel(output={"passed": True}), _entry())

    assert (ok, error) == (True, None)


def test_a_mismatched_case_fails_without_being_treated_as_an_error():
    # Wrong, but not raised - must be told apart from a real failure so
    # _print_output_shape_suggestion doesn't fire from ordinary wrong answers.
    ok, error, _ = _run(_FakeModel(output={"passed": False}), _entry())

    assert ok is False
    assert error is None


def test_a_matching_case_sets_the_entrys_model_output():
    entry = _entry()

    _run(_FakeModel(output={"passed": True}), entry)

    assert entry.get_model_output().get_output() == {"passed": True}


# --------------------------------------------------------------------------
# run_case: progress reporting - the case position/id/chain feed the
# wait-bar's prefix, and every case (pass, fail, or error) records into the
# shared tracker.
# --------------------------------------------------------------------------


def test_run_case_records_into_the_given_tracker():
    tracker = ProgressTracker(1)

    run_case(_FakeModel(output={"passed": True}), _entry(), 1, 1, tracker)

    assert tracker.get_done() == 1


def test_run_case_builds_a_prefix_with_the_position_id_and_chain_type(capsys):
    run_case(_FakeModel(output={"passed": True}), _entry("case-9"), 3, 5, ProgressTracker(1))

    printed = capsys.readouterr().out
    assert "case-9" in printed
    assert "(3)/(5)" in printed
    assert "fake-chain" in printed


# --------------------------------------------------------------------------
# _print_output_shape_suggestion: only fires on a 100% failure rate.
# --------------------------------------------------------------------------


def test_stays_silent_when_not_every_case_failed(capsys):
    _print_output_shape_suggestion(failed=1, total=2, last_error=RuntimeError("x"))

    assert capsys.readouterr().out == ""


def test_stays_silent_for_an_empty_corpus(capsys):
    _print_output_shape_suggestion(failed=0, total=0, last_error=None)

    assert capsys.readouterr().out == ""


def test_names_the_last_error_when_every_case_failed(capsys):
    _print_output_shape_suggestion(failed=2, total=2, last_error=RuntimeError("bad reply"))

    printed = capsys.readouterr().out
    assert "every case failed to run" in printed
    assert "RuntimeError: bad reply" in printed


# --------------------------------------------------------------------------
# _resolve_check_dir: --continue lands in the exact existing check_N dir that
# was made with this run's own prompt and chain config - never in one made
# with another setup, which would have its cached predictions wiped and its
# report overwritten. Same guarantee trainingLoop's --continue gives via
# resume.resumable_run_number().
# --------------------------------------------------------------------------


class _FakeCheckDirModel:
    def __init__(self, tier: str = "cheap", prompt: str = "a prompt") -> None:
        self._tier = tier
        self._prompt = prompt

    def get_type(self) -> str:
        return "tonality"

    def get_chain_type(self) -> str:
        return "template_checklist"

    def get_tier(self) -> str:
        return self._tier

    def get_config(self) -> dict:
        return {"type": "tonality", "chain": "template_checklist", "tier": self._tier}

    def get_system_prompt(self) -> str:
        return self._prompt


def _check_dir_path(*parts: str) -> str:
    # cwd-relative, like find_corpus_path() - see test_report.py's write_report tests.
    return os.path.join("workflows", "tonality", ".temp", "template_checklist", *parts)


def _make_check_dir(tmp_path, name: str, prompt: str = "a prompt", tier: str = "cheap") -> None:
    check_dir = tmp_path / "workflows" / "tonality" / ".temp" / "template_checklist" / name
    check_dir.mkdir(parents=True)
    (check_dir / "prompt.txt").write_text(prompt)
    (check_dir / "config.json").write_text(
        json.dumps({"type": "tonality", "chain": "template_checklist", "tier": tier})
    )


def test_without_continue_it_always_returns_the_next_fresh_check_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _make_check_dir(tmp_path, "check_0_cheap")

    assert _resolve_check_dir(_FakeCheckDirModel(), continue_run=False) == _check_dir_path(
        "check_1_cheap"
    )


def test_with_continue_but_no_previous_run_it_falls_back_to_a_fresh_check_dir(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)

    result = _resolve_check_dir(_FakeCheckDirModel(), continue_run=True)

    assert result == _check_dir_path("check_0_cheap")
    assert "no previous run exists" in capsys.readouterr().out


def test_with_continue_it_returns_the_exact_existing_check_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _make_check_dir(tmp_path, "check_0_cheap")

    assert _resolve_check_dir(_FakeCheckDirModel(), continue_run=True) == _check_dir_path(
        "check_0_cheap"
    )


def test_with_continue_it_keeps_the_suffix_a_freshly_built_dir_would_get_wrong(
    monkeypatch, tmp_path
):
    # The dir's *name* says "other-tier" while its config.json says this
    # model's own tier - the name is cosmetic, so --continue must land in the
    # real directory, not one built from the model's current tier.
    monkeypatch.chdir(tmp_path)
    _make_check_dir(tmp_path, "check_0_other-tier")

    assert _resolve_check_dir(_FakeCheckDirModel(), continue_run=True) == _check_dir_path(
        "check_0_other-tier"
    )


def test_continue_with_a_different_tier_starts_a_new_run_beside_the_old_one(
    monkeypatch, tmp_path, capsys
):
    # The reported bug: this used to return check_0_thinking, whose cached
    # predictions LoopCache then wiped and whose report was overwritten -
    # in a folder still named check_0_thinking.
    monkeypatch.chdir(tmp_path)
    _make_check_dir(tmp_path, "check_0_thinking", tier="thinking")

    result = _resolve_check_dir(_FakeCheckDirModel(tier="fast"), continue_run=True)

    assert result == _check_dir_path("check_1_fast")
    assert os.path.isfile(_check_dir_path("check_0_thinking", "config.json"))


def test_continue_with_a_different_prompt_starts_a_new_run_too(monkeypatch, tmp_path):
    # Same tier, so the directory name matches - only the saved prompt tells
    # the two runs apart. --prompt-file needs the same protection.
    monkeypatch.chdir(tmp_path)
    _make_check_dir(tmp_path, "check_0_cheap", prompt="the chain's own prompt")

    result = _resolve_check_dir(_FakeCheckDirModel(prompt="a candidate prompt"), continue_run=True)

    assert result == _check_dir_path("check_1_cheap")


def test_continue_says_why_it_skipped_the_runs_it_found(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    _make_check_dir(tmp_path, "check_0_thinking", tier="thinking")

    _resolve_check_dir(_FakeCheckDirModel(tier="fast"), continue_run=True)

    printed = capsys.readouterr().out
    assert "none of the 1 existing run(s)" in printed
    assert "check_0_thinking" in printed
    assert "tier 'fast'" in printed
    assert "Leaving them untouched" in printed


# --------------------------------------------------------------------------
# _save_failed_case: a case that raised has no prediction to cache. Its
# transcript is kept for reading, beside the cache rather than in it, so the
# next --continue retries the case instead of recalling an empty answer as a
# real one. Same rule as ClassificationLoop._run_entry, which saves nothing.
# --------------------------------------------------------------------------


def _failed_entry(entry_id="a", convo="the raw prompt and reply") -> Entry:
    entry = _entry(entry_id)
    if convo:
        entry.set_model_output(ModelOutput(convo, {}, {}))
    return entry


def _cache_for(tmp_path, *entries) -> LoopCache:
    corpus = Corpus()
    for entry in entries:
        corpus.add_entry(entry)
    return LoopCache("check_0_cheap", corpus, _FakeCheckDirModel(), base_dir=str(tmp_path))


def test_a_failed_case_is_not_written_into_the_cache(tmp_path):
    entry = _failed_entry()
    cache = _cache_for(tmp_path, entry)

    _save_failed_case(cache, entry)

    assert not os.path.exists(cache.get_entry_path("a"))


def test_a_failed_cases_transcript_is_kept_for_reading(tmp_path):
    entry = _failed_entry()
    cache = _cache_for(tmp_path, entry)

    _save_failed_case(cache, entry)

    assert os.path.isfile(cache.get_failure_path("a"))
    with open(cache.get_failure_path("a")) as f:
        assert "the raw prompt and reply" in f.read()


def test_a_failed_case_is_retried_rather_than_recalled(tmp_path):
    # The reported bug: the failure used to be saved under entries/, where
    # recall() read its {} output back as a real answer - the case was never
    # retried, scored as a wrong judgment, and still counted as 'parsed'.
    entry = _failed_entry()
    cache = _cache_for(tmp_path, entry)
    cache.save_prompt()
    cache.save_config()

    _save_failed_case(cache, entry)

    assert cache.recall() == set()


def test_a_failure_transcript_does_not_disturb_the_cases_that_did_answer(tmp_path):
    answered, failed = _entry("a"), _failed_entry("b")
    cache = _cache_for(tmp_path, answered, failed)
    cache.save_prompt()
    cache.save_config()
    cache.save_prediction("a", ModelOutput("convo", {"passed": True}, {}))

    _save_failed_case(cache, failed)

    assert cache.recall() == {"a"}


def test_a_failed_case_with_nothing_to_show_writes_no_file(tmp_path):
    # A plain exception leaves no calls to render (see
    # _record_failed_conversation), so there is nothing worth a file.
    entry = _failed_entry(convo="")
    cache = _cache_for(tmp_path, entry)

    _save_failed_case(cache, entry)

    assert not os.path.exists(cache.get_failure_path("a"))


def test_saving_a_failed_case_says_where_it_went_and_why(tmp_path, capsys):
    entry = _failed_entry()
    cache = _cache_for(tmp_path, entry)

    _save_failed_case(cache, entry)

    printed = capsys.readouterr().out
    assert os.path.join("failures", "a.txt") in printed
    assert "--continue retries this case" in printed


# --------------------------------------------------------------------------
# _score_recalled: scores a recalled entry the same way run_case scores a
# freshly-predicted one - matches() against whatever's already on the entry.
# --------------------------------------------------------------------------


def test_score_recalled_is_true_when_the_recalled_output_matches():
    entry = _entry(expected={"passed": True})
    entry.set_model_output(ModelOutput("convo", {"passed": True}))

    assert _score_recalled(entry) is True


def test_score_recalled_is_false_when_the_recalled_output_mismatches():
    entry = _entry(expected={"passed": True})
    entry.set_model_output(ModelOutput("convo", {"passed": False}))

    assert _score_recalled(entry) is False
