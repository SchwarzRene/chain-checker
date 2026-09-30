import json

from chain_checker.baseclasses.metrics.names import modifier_run_info
from chain_checker.utils import metrics_file
from chain_checker.utils.training_utils import metrics_io


def test_load_epoch_metrics_reads_every_epoch_that_has_a_metrics_json(tmp_path):
    (tmp_path / "modification_0").mkdir()
    (tmp_path / "modification_0" / "metrics.json").write_text('{"accuracy": 0.5}')
    # Epoch 1 ran but hasn't saved metrics.json yet - it must not show up.
    (tmp_path / "modification_1").mkdir()

    assert metrics_io.load_epoch_metrics(str(tmp_path)) == {"modification_0": {"accuracy": 0.5}}


def test_load_previous_metrics_returns_an_empty_dict_when_the_file_is_missing(tmp_path):
    assert metrics_io.load_previous_metrics(str(tmp_path)) == {}


def test_load_previous_metrics_returns_an_empty_dict_for_corrupt_json(tmp_path):
    (tmp_path / "metrics.json").write_text("{not valid json")

    assert metrics_io.load_previous_metrics(str(tmp_path)) == {}


def test_load_previous_metrics_returns_the_parsed_contents(tmp_path):
    (tmp_path / "metrics.json").write_text('{"accuracy": 0.9}')

    assert metrics_io.load_previous_metrics(str(tmp_path)) == {"accuracy": 0.9}


def test_serialise_metrics_round_trips_through_json():
    data = {"accuracy": 0.5, "nested": {"a": 1}}

    assert json.loads(metrics_file.serialise_metrics(data)) == data


def test_reconstruct_metrics_list_keeps_only_the_real_metric_entries():
    run_report_data = {
        "Accuracy-Metrics": {"accuracy": 0.5},
        "prompt": "some prompt",
        "run_info": {"chain": "x"},
        "val": {},
        "final_validation_only": True,
        modifier_run_info: {"tier": "cheap"},
    }

    result = metrics_io.reconstruct_metrics_list(run_report_data)

    assert result == [
        {
            "name": "Accuracy-Metrics",
            "description": "(historical run - see this epoch's own report.html)",
            "results": {"accuracy": 0.5},
        }
    ]
