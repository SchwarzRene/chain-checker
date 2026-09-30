import os

from chain_checker.utils.training_utils import paths


def test_run_dir_and_epoch_dir_nest_under_the_base_dir():
    assert paths.run_dir("base", 2) == os.path.join("base", "run_2")
    assert paths.epoch_dir(os.path.join("base", "run_2"), 3) == os.path.join(
        "base", "run_2", "modification_3"
    )


def test_run_dir_appends_a_tier_as_a_cosmetic_suffix_for_a_new_run(tmp_path):
    assert paths.run_dir(str(tmp_path), 0, tier="balanced") == str(tmp_path / "run_0_balanced")


def test_run_dir_slugifies_an_unsafe_tier(tmp_path):
    assert paths.run_dir(str(tmp_path), 0, tier="qwen3.5:0.8b") == str(
        tmp_path / "run_0_qwen3.5-0.8b"
    )


def test_run_dir_reuses_the_exact_existing_suffixed_dir_regardless_of_tier(tmp_path):
    # --continue must land in the run it already has - not a fresh guess
    # built from whatever tier the current invocation happens to pass. Which
    # run may be continued at all is decided before this, by
    # resume.resumable_run_number(); by the time a number reaches run_dir()
    # it is either vetted or brand new, so resolving it by number alone is
    # safe here.
    (tmp_path / "run_0_balanced").mkdir()

    assert paths.run_dir(str(tmp_path), 0, tier="fast") == str(tmp_path / "run_0_balanced")
    assert paths.run_dir(str(tmp_path), 0) == str(tmp_path / "run_0_balanced")


def test_existing_run_numbers_is_empty_for_a_dir_that_does_not_exist(tmp_path):
    assert paths.existing_run_numbers(str(tmp_path / "missing")) == []


def test_existing_run_numbers_finds_and_sorts_run_dirs(tmp_path):
    (tmp_path / "run_2").mkdir()
    (tmp_path / "run_0").mkdir()
    (tmp_path / "run_1").mkdir()
    (tmp_path / "not_a_run").mkdir()

    assert paths.existing_run_numbers(str(tmp_path)) == [0, 1, 2]


def test_existing_run_numbers_counts_a_tier_suffixed_dir_by_its_number(tmp_path):
    (tmp_path / "run_0_balanced").mkdir()
    (tmp_path / "run_1_qwen3.5-0.8b").mkdir()

    assert paths.existing_run_numbers(str(tmp_path)) == [0, 1]


def test_epoch_names_is_empty_for_a_dir_that_does_not_exist(tmp_path):
    assert paths.epoch_names(str(tmp_path / "missing")) == []


def test_epoch_names_sorts_numerically_not_lexically(tmp_path):
    (tmp_path / "modification_10").mkdir()
    (tmp_path / "modification_2").mkdir()
    (tmp_path / "summary").mkdir()

    assert paths.epoch_names(str(tmp_path)) == ["modification_2", "modification_10"]


def test_final_epoch_done_is_false_without_a_metrics_json(tmp_path):
    assert paths.final_epoch_done(str(tmp_path), 0) is False


def test_final_epoch_done_is_true_once_the_epochs_metrics_json_exists(tmp_path):
    epoch_dir = tmp_path / "modification_0"
    epoch_dir.mkdir()
    (epoch_dir / "metrics.json").write_text("{}")

    assert paths.final_epoch_done(str(tmp_path), 0) is True
