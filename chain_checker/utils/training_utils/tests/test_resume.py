import json
import os

from chain_checker.utils.training_utils import resume


class _FakeModel:
    def __init__(self, tier: str = "cheap") -> None:
        self.prompt = None
        self._tier = tier

    def set_new_system_prompt(self, prompt: str) -> None:
        self.prompt = prompt

    def get_config(self) -> dict:
        return {"type": "tonality", "chain": "template_checklist", "tier": self._tier}


class _FakeModifier:
    def __init__(self) -> None:
        self.replayed = []

    def replay_run(self, prompt, data) -> None:
        self.replayed.append((prompt, data))


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# --------------------------------------------------------------------------
# resume_epoch
# --------------------------------------------------------------------------


def test_a_fresh_run_dir_with_no_completed_epochs_resumes_at_zero(tmp_path):
    model = _FakeModel()

    assert resume.resume_epoch(model, str(tmp_path)) == 0
    assert model.prompt is None


def test_resumes_after_the_last_epoch_that_has_a_modifier_new_prompt(tmp_path):
    run_dir = str(tmp_path)
    _write(os.path.join(run_dir, "modification_0", "modifier_new_prompt.txt"), "prompt after 0")
    model = _FakeModel()

    epoch = resume.resume_epoch(model, run_dir)

    assert epoch == 1
    assert model.prompt == "prompt after 0"


def test_prefers_the_resuming_epochs_own_prompt_over_the_previous_epochs_rewrite(tmp_path):
    run_dir = str(tmp_path)
    _write(os.path.join(run_dir, "modification_0", "modifier_new_prompt.txt"), "prompt after 0")
    _write(os.path.join(run_dir, "modification_1", "prompt.txt"), "epoch 1's own prompt")
    model = _FakeModel()

    epoch = resume.resume_epoch(model, run_dir)

    assert epoch == 1
    assert model.prompt == "epoch 1's own prompt"


# --------------------------------------------------------------------------
# replay_modifier_history
# --------------------------------------------------------------------------


def test_replays_every_epoch_that_has_both_a_prompt_and_metrics(tmp_path):
    run_dir = str(tmp_path)
    _write(os.path.join(run_dir, "modification_0", "prompt.txt"), "prompt 0")
    _write(
        os.path.join(run_dir, "modification_0", "metrics.json"),
        '{"Accuracy-Metrics": {"accuracy": 0.5}}',
    )
    modifier = _FakeModifier()

    resume.replay_modifier_history(modifier, run_dir, upto_epoch=1)

    assert len(modifier.replayed) == 1
    prompt, data = modifier.replayed[0]
    assert prompt == "prompt 0"
    assert data == [
        {
            "name": "Accuracy-Metrics",
            "description": "(historical run - see this epoch's own report.html)",
            "results": {"accuracy": 0.5},
        }
    ]


def test_skips_an_epoch_missing_a_prompt_or_metrics_file(tmp_path):
    run_dir = str(tmp_path)
    # Epoch 0 has metrics but no prompt.txt (never fully finished).
    _write(os.path.join(run_dir, "modification_0", "metrics.json"), "{}")
    modifier = _FakeModifier()

    resume.replay_modifier_history(modifier, run_dir, upto_epoch=1)

    assert modifier.replayed == []


# --------------------------------------------------------------------------
# resumable_run_number: --continue may only add epochs to a run trained
# against the same chain config - never to another tier's run, which would
# leave one run_N whose epochs were trained against two different models.
# --------------------------------------------------------------------------


def _make_run(tmp_path, name: str, tier: str = "cheap", epochs: int = 1) -> None:
    """A run as the training loop leaves it: one config.json per epoch,
    written by the epoch's own LoopCache before its first prediction."""
    for epoch in range(epochs):
        _write(
            os.path.join(str(tmp_path), name, f"modification_{epoch}", "config.json"),
            json.dumps({"type": "tonality", "chain": "template_checklist", "tier": tier}),
        )


def test_no_runs_on_disk_means_nothing_to_resume(tmp_path):
    assert resume.resumable_run_number(str(tmp_path), _FakeModel()) is None


def test_the_newest_run_with_a_matching_config_is_resumed(tmp_path):
    _make_run(tmp_path, "run_0_cheap")
    _make_run(tmp_path, "run_1_cheap")

    assert resume.resumable_run_number(str(tmp_path), _FakeModel()) == 1


def test_a_run_trained_against_another_tier_is_not_resumed(tmp_path):
    _make_run(tmp_path, "run_0_thinking", tier="thinking")

    assert resume.resumable_run_number(str(tmp_path), _FakeModel(tier="fast")) is None


def test_an_older_matching_run_is_resumed_past_a_newer_foreign_one(tmp_path):
    _make_run(tmp_path, "run_0_cheap")
    _make_run(tmp_path, "run_1_thinking", tier="thinking")

    assert resume.resumable_run_number(str(tmp_path), _FakeModel()) == 0


def test_only_the_first_epochs_config_decides_it(tmp_path):
    # Every epoch of a run shares one chain config, so epoch 0 speaks for the
    # whole run - unlike the prompt, which is meant to change each epoch.
    _make_run(tmp_path, "run_0_cheap", epochs=3)

    assert resume.resumable_run_number(str(tmp_path), _FakeModel()) == 0


def test_an_empty_run_dir_is_resumable(tmp_path):
    # write_summary() creates the run dir before epoch 0, so an interrupted
    # start leaves a shell with no epochs - and no foreign results to protect.
    (tmp_path / "run_0_cheap").mkdir()

    assert resume.resumable_run_number(str(tmp_path), _FakeModel(tier="fast")) == 0
