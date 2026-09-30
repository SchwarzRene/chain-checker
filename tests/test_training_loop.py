import argparse

import pytest

import chain_checker.trainingLoop as tl


class _FakeModel:
    def get_prompt_template_vars(self):
        return ()

    def get_tier(self):
        return "test-tier"

    def get_type(self):
        return "test-type"

    def get_chain_type(self):
        return "test-chain"


# --------------------------------------------------------------------------
# _build_modifier: picks the modifier's backend based on --modifier-backend.
# --------------------------------------------------------------------------


def test_ollama_backend_is_initialized_with_the_modifier_model_name():
    args = argparse.Namespace(
        modifier_backend="ollama",
        modifier_model="qwen3.5:4b",
        modifier_app="unused",
        modifier_tier="unused",
    )

    modifier = tl._build_modifier(_FakeModel(), args)

    assert modifier.get_run_info() == {"backend": "ollama", "model": "qwen3.5:4b"}


def test_litellm_backend_is_initialized_with_the_app_and_tier(monkeypatch):
    calls = []
    monkeypatch.setattr(
        tl.ModifierModel,
        "init_litellm",
        lambda self, app_label, tier: calls.append((app_label, tier)),
    )
    args = argparse.Namespace(
        modifier_backend="litellm",
        modifier_model="unused",
        modifier_app="tonality",
        modifier_tier="thinking",
    )

    tl._build_modifier(_FakeModel(), args)

    assert calls == [("tonality", "thinking")]


def test_any_non_ollama_backend_string_still_uses_litellm(monkeypatch):
    # Only "ollama" is special-cased - argparse already restricts the flag to
    # the two real backends, so anything else here means "litellm".
    calls = []
    monkeypatch.setattr(
        tl.ModifierModel, "init_litellm", lambda self, app_label, tier: calls.append(True)
    )
    args = argparse.Namespace(
        modifier_backend="litellm", modifier_model="unused", modifier_app="a", modifier_tier="fast"
    )

    tl._build_modifier(_FakeModel(), args)

    assert calls == [True]


# --------------------------------------------------------------------------
# _print_startup_banner: every tracked value and its source, printed once.
# --------------------------------------------------------------------------


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        type="tonality",
        chain_type="template_checklist",
        chain_tier=None,
        epochs=4,
        modifier_backend="litellm",
        modifier_model="qwen3.5:4b",
        modifier_app="tonality",
        modifier_tier="fast",
        continue_run=False,
        val_file=None,
        prompt_file=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _sources(**overrides) -> dict:
    base = {
        key: "default"
        for key in (
            "type",
            "chain_type",
            "chain_tier",
            "file",
            "val_file",
            "prompt_file",
            "epochs",
            "modifier_backend",
            "modifier_model",
            "modifier_app",
            "modifier_tier",
            "continue_run",
        )
    }
    base.update(overrides)
    return base


def test_banner_prints_every_tracked_value_and_its_source(capsys):
    tl._print_startup_banner(_args(), _sources(), "workflows/tonality/corpus.yaml", None)

    printed = capsys.readouterr().out
    assert "type: 'tonality' (default)" in printed
    assert "chain-type: 'template_checklist' (default)" in printed
    assert "chain-tier: chain's own default (default)" in printed
    assert "file: 'workflows/tonality/corpus.yaml' (default)" in printed
    assert "val-file: none (default)" in printed
    assert "prompt-file: none (default)" in printed
    assert "epochs: 4 (default)" in printed
    assert "modifier-backend: 'litellm' (default)" in printed
    assert "modifier-model: 'qwen3.5:4b' (default)" in printed
    assert "modifier-app: 'tonality' (default)" in printed
    assert "modifier-tier: 'fast' (default)" in printed
    assert "continue: False (default)" in printed


def test_banner_shows_the_config_path_only_when_one_was_used(capsys):
    tl._print_startup_banner(_args(), _sources(), "c.yaml", None)
    assert "config:" not in capsys.readouterr().out

    tl._print_startup_banner(_args(), _sources(), "c.yaml", "my_run.yaml")
    assert "config: 'my_run.yaml'" in capsys.readouterr().out


def test_banner_shows_val_file_and_prompt_file_when_given(capsys):
    args = _args(val_file="val.yaml", prompt_file="candidate.txt")

    tl._print_startup_banner(args, _sources(), "c.yaml", None)

    printed = capsys.readouterr().out
    assert "val-file: 'val.yaml'" in printed
    assert "prompt-file: 'candidate.txt'" in printed


# --------------------------------------------------------------------------
# _load_corpora: the training corpus, plus an optional held-out val corpus.
# --------------------------------------------------------------------------


def _write_corpus(path, case_id="a") -> None:
    path.write_text(
        f"cases:\n  - id: {case_id}\n    input:\n      text: hi\n    output:\n      passed: true\n"
    )


def test_load_corpora_loads_just_the_training_corpus_when_no_val_file(tmp_path):
    corpus_path = tmp_path / "corpus.yaml"
    _write_corpus(corpus_path)

    corpus, val_corpus = tl._load_corpora(str(corpus_path), None)

    assert len(corpus) == 1
    assert val_corpus is None


def test_load_corpora_also_loads_the_val_file_when_given(tmp_path):
    corpus_path = tmp_path / "corpus.yaml"
    val_path = tmp_path / "val.yaml"
    _write_corpus(corpus_path, "a")
    _write_corpus(val_path, "b")

    corpus, val_corpus = tl._load_corpora(str(corpus_path), str(val_path))

    assert len(corpus) == 1
    assert len(val_corpus) == 1
    assert val_corpus.get_by_id("b") is not None


# --------------------------------------------------------------------------
# loop(): orchestration - run numbering, --continue resume, epoch iteration,
# the final validation pass, and the summary/overview writes. Every
# collaborator (paths, reports, resume, run_epoch) is faked so this only
# exercises loop()'s own control flow.
# --------------------------------------------------------------------------


@pytest.fixture
def loop_recorder(monkeypatch):
    calls = {
        "run_epoch": [],
        "write_summary": [],
        "resume_epoch_called_with": None,
        "resume_epoch_return": 0,
        "resumable_run_return": None,
        "replay": [],
        "overview_return": None,
    }

    def fake_run_epoch(
        epoch, total_epochs, corpus, model, modifier, run_dir, ask_for_rewrite, val_corpus=None
    ):
        calls["run_epoch"].append(
            {
                "epoch": epoch,
                "total_epochs": total_epochs,
                "run_dir": run_dir,
                "ask_for_rewrite": ask_for_rewrite,
            }
        )
        return {}

    def fake_write_summary(run_dir):
        calls["write_summary"].append(run_dir)
        return f"{run_dir}/summary/report.html"

    def fake_write_runs_overview(model):
        return calls["overview_return"]

    def fake_resume_epoch(model, run_dir):
        calls["resume_epoch_called_with"] = run_dir
        return calls["resume_epoch_return"]

    def fake_resumable_run_number(base_dir, model):
        return calls["resumable_run_return"]

    def fake_replay(modifier, run_dir, upto_epoch):
        calls["replay"].append((run_dir, upto_epoch))

    monkeypatch.setattr(tl, "run_epoch", fake_run_epoch)
    monkeypatch.setattr(tl.reports, "write_summary", fake_write_summary)
    monkeypatch.setattr(tl.reports, "write_runs_overview", fake_write_runs_overview)
    monkeypatch.setattr(tl.resume, "resume_epoch", fake_resume_epoch)
    monkeypatch.setattr(tl.resume, "resumable_run_number", fake_resumable_run_number)
    monkeypatch.setattr(tl.resume, "replay_modifier_history", fake_replay)
    monkeypatch.setattr(tl, "default_base_dir", lambda type, chain_type: "BASE")
    monkeypatch.setattr(tl.paths, "run_dir", lambda base_dir, n, tier=None: f"{base_dir}/run_{n}")
    monkeypatch.setattr(tl.paths, "final_epoch_done", lambda run_dir, epoch: False)
    monkeypatch.setattr(tl.paths, "existing_run_numbers", lambda base_dir: [])

    return calls


def _run_loop(epochs=2, continue_run=False):
    tl.loop(epochs, object(), _FakeModel(), object(), continue_run=continue_run)


def test_a_fresh_run_starts_at_run_0_when_none_exist(monkeypatch, loop_recorder):
    monkeypatch.setattr(tl.paths, "existing_run_numbers", lambda base_dir: [])

    _run_loop()

    assert {c["run_dir"] for c in loop_recorder["run_epoch"]} == {"BASE/run_0"}


def test_a_fresh_run_continues_the_run_numbering_past_existing_runs(monkeypatch, loop_recorder):
    monkeypatch.setattr(tl.paths, "existing_run_numbers", lambda base_dir: [0, 1])

    _run_loop(epochs=1)

    assert {c["run_dir"] for c in loop_recorder["run_epoch"]} == {"BASE/run_2"}


def test_every_training_epoch_runs_with_ask_for_rewrite(loop_recorder):
    _run_loop(epochs=3)

    training_epochs = [c["epoch"] for c in loop_recorder["run_epoch"] if c["ask_for_rewrite"]]
    assert training_epochs == [0, 1, 2]


def test_a_final_validation_pass_runs_after_training_epochs(loop_recorder):
    _run_loop(epochs=2)

    final = [c for c in loop_recorder["run_epoch"] if not c["ask_for_rewrite"]]
    assert len(final) == 1
    assert final[0]["epoch"] == 2


def test_the_final_pass_is_skipped_when_already_done(monkeypatch, loop_recorder, capsys):
    monkeypatch.setattr(tl.paths, "final_epoch_done", lambda run_dir, epoch: True)

    _run_loop(epochs=2)

    assert all(c["ask_for_rewrite"] for c in loop_recorder["run_epoch"])
    assert "already done" in capsys.readouterr().out


def test_a_summary_is_written_before_every_epoch_and_after_the_final_pass(loop_recorder):
    _run_loop(epochs=2)

    # one write before the epoch loop starts, one after each of the 2
    # training epochs, and one after the final validation pass
    assert loop_recorder["write_summary"] == ["BASE/run_0"] * 4


def test_no_extra_summary_write_when_the_final_pass_is_skipped(monkeypatch, loop_recorder):
    monkeypatch.setattr(tl.paths, "final_epoch_done", lambda run_dir, epoch: True)

    _run_loop(epochs=2)

    # one write before the loop, one after each of the 2 training epochs -
    # none for the skipped final pass
    assert loop_recorder["write_summary"] == ["BASE/run_0"] * 3


def test_continue_resumes_from_the_newest_resumable_run(monkeypatch, loop_recorder):
    monkeypatch.setattr(tl.paths, "existing_run_numbers", lambda base_dir: [0, 1])
    loop_recorder["resumable_run_return"] = 1
    loop_recorder["resume_epoch_return"] = 2

    _run_loop(epochs=4, continue_run=True)

    assert loop_recorder["resume_epoch_called_with"] == "BASE/run_1"
    training_epochs = [c["epoch"] for c in loop_recorder["run_epoch"] if c["ask_for_rewrite"]]
    assert training_epochs == [2, 3]


def test_continue_replays_the_modifiers_history_when_resuming_mid_run(monkeypatch, loop_recorder):
    monkeypatch.setattr(tl.paths, "existing_run_numbers", lambda base_dir: [0])
    loop_recorder["resumable_run_return"] = 0
    loop_recorder["resume_epoch_return"] = 2

    _run_loop(epochs=4, continue_run=True)

    assert loop_recorder["replay"] == [("BASE/run_0", 2)]


def test_continue_does_not_replay_history_when_starting_at_epoch_0(monkeypatch, loop_recorder):
    monkeypatch.setattr(tl.paths, "existing_run_numbers", lambda base_dir: [0])
    loop_recorder["resumable_run_return"] = 0
    loop_recorder["resume_epoch_return"] = 0

    _run_loop(epochs=1, continue_run=True)

    assert loop_recorder["replay"] == []


def test_continue_with_no_previous_run_starts_fresh_at_run_0_and_warns(
    monkeypatch, loop_recorder, capsys
):
    monkeypatch.setattr(tl.paths, "existing_run_numbers", lambda base_dir: [])

    _run_loop(epochs=1, continue_run=True)

    assert {c["run_dir"] for c in loop_recorder["run_epoch"]} == {"BASE/run_0"}
    # Nothing to restore from a directory that does not exist yet.
    assert loop_recorder["resume_epoch_called_with"] is None
    assert "no previous run exists" in capsys.readouterr().out


def test_continue_starts_a_new_run_when_no_existing_run_is_resumable(
    monkeypatch, loop_recorder, capsys
):
    # Every run on disk was trained against another chain config (another
    # tier): --continue must number past them rather than append this tier's
    # epochs to one of them.
    monkeypatch.setattr(tl.paths, "existing_run_numbers", lambda base_dir: [0, 1])
    loop_recorder["resumable_run_return"] = None

    _run_loop(epochs=2, continue_run=True)

    assert {c["run_dir"] for c in loop_recorder["run_epoch"]} == {"BASE/run_2"}
    assert loop_recorder["resume_epoch_called_with"] is None
    training_epochs = [c["epoch"] for c in loop_recorder["run_epoch"] if c["ask_for_rewrite"]]
    assert training_epochs == [0, 1]


def test_continue_says_why_it_skipped_the_runs_it_found(monkeypatch, loop_recorder, capsys):
    monkeypatch.setattr(tl.paths, "existing_run_numbers", lambda base_dir: [0, 1])
    loop_recorder["resumable_run_return"] = None

    _run_loop(epochs=1, continue_run=True)

    printed = capsys.readouterr().out
    assert "none of the 2 existing run(s)" in printed
    assert "tier 'test-tier'" in printed
    assert "mix two tiers into a single run" in printed


def test_the_overview_is_reported_once_written(loop_recorder, capsys):
    loop_recorder["overview_return"] = "BASE/runs_overview.html"

    _run_loop(epochs=1)

    assert "comparison written to BASE/runs_overview.html" in capsys.readouterr().out


def test_no_overview_message_when_nothing_was_written(loop_recorder, capsys):
    _run_loop(epochs=1)

    assert "comparison written to" not in capsys.readouterr().out
