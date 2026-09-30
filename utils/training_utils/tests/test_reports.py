import json
import os

from chain_checker.utils.training_utils import reports


class _FakeModel:
    def get_type(self) -> str:
        return "tonality"

    def get_chain_type(self) -> str:
        return "template_checklist"


def _write_epoch_metrics(run_dir: str, epoch_name: str, accuracy: float, **extra) -> None:
    epoch_dir = os.path.join(run_dir, epoch_name)
    os.makedirs(epoch_dir, exist_ok=True)
    with open(os.path.join(epoch_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"Accuracy-Metrics": {"accuracy": accuracy}, "prompt": "p", **extra}, f)


# --------------------------------------------------------------------------
# write_summary
# --------------------------------------------------------------------------


def test_write_summary_creates_a_report_under_a_summary_subdir(tmp_path):
    run_dir = str(tmp_path / "run_0")
    _write_epoch_metrics(run_dir, "modification_0", 0.5)

    report_path = reports.write_summary(run_dir)

    assert report_path == os.path.join(run_dir, "summary", "report.html")
    assert os.path.isfile(report_path)


def test_write_summary_works_even_with_no_epochs_yet(tmp_path):
    run_dir = str(tmp_path / "run_0")
    os.makedirs(run_dir)

    report_path = reports.write_summary(run_dir)

    assert os.path.isfile(report_path)


# --------------------------------------------------------------------------
# write_runs_overview
# --------------------------------------------------------------------------


def test_write_runs_overview_returns_none_with_fewer_than_two_runs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    base_dir = os.path.join("workflows", "tonality", ".temp", "template_checklist")
    _write_epoch_metrics(os.path.join(base_dir, "run_0"), "modification_0", 0.5)

    assert reports.write_runs_overview(_FakeModel()) is None


def test_write_runs_overview_compares_the_latest_epoch_of_each_run(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    base_dir = os.path.join("workflows", "tonality", ".temp", "template_checklist")
    _write_epoch_metrics(os.path.join(base_dir, "run_0"), "modification_0", 0.5)
    _write_epoch_metrics(os.path.join(base_dir, "run_1"), "modification_0", 0.6)
    _write_epoch_metrics(os.path.join(base_dir, "run_1"), "modification_1", 0.8)

    overview_path = reports.write_runs_overview(_FakeModel())

    assert overview_path == os.path.join(base_dir, "runs_overview.html")
    assert os.path.isfile(overview_path)


def test_write_runs_overview_never_shows_mispredicted_entries(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    base_dir = os.path.join("workflows", "tonality", ".temp", "template_checklist")
    mispredictions = {
        "Negative-Predicted-Metrics": {
            "case-3": {
                "input": {"text": "hello"},
                "true": {"passed": True},
                "predicted": {"passed": False},
                "matches": {"passed": False},
            }
        }
    }
    _write_epoch_metrics(os.path.join(base_dir, "run_0"), "modification_0", 0.5, **mispredictions)
    _write_epoch_metrics(os.path.join(base_dir, "run_1"), "modification_0", 0.6, **mispredictions)

    overview_path = reports.write_runs_overview(_FakeModel())

    with open(overview_path, encoding="utf-8") as f:
        overview_html = f.read()
    assert "Mispredicted entries" not in overview_html
    assert "case-3" not in overview_html


def test_write_runs_overview_collapses_each_runs_prompt(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    base_dir = os.path.join("workflows", "tonality", ".temp", "template_checklist")
    _write_epoch_metrics(os.path.join(base_dir, "run_0"), "modification_0", 0.5)
    _write_epoch_metrics(os.path.join(base_dir, "run_1"), "modification_0", 0.6)

    overview_path = reports.write_runs_overview(_FakeModel())

    with open(overview_path, encoding="utf-8") as f:
        overview_html = f.read()
    assert '<details class="prompt-entry">' in overview_html
    assert '<div class="card prompt-card">' not in overview_html
