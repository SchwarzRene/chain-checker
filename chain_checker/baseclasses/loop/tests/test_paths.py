import os

import pytest

from chain_checker.baseclasses.loop.paths import (
    RunPaths,
    default_base_dir,
    entry_id_from_filename,
    slugify_tier,
)


@pytest.fixture
def paths() -> RunPaths:
    return RunPaths("modification_0", os.path.join("base", "dir"))


def test_the_run_dir_is_the_run_id_under_the_base_dir(paths):
    assert paths.run_dir() == os.path.join("base", "dir", "modification_0")


def test_the_three_files_of_a_run_sit_beside_each_other(paths):
    assert paths.prompt_file() == os.path.join(paths.run_dir(), "prompt.txt")
    assert paths.config_file() == os.path.join(paths.run_dir(), "config.json")
    assert paths.entries_dir() == os.path.join(paths.run_dir(), "entries")


def test_an_entry_is_a_text_file_named_after_its_id(paths):
    assert paths.entry_file("case-1") == os.path.join(paths.entries_dir(), "case-1.txt")


def test_an_int_id_names_a_file_the_same_way_a_str_one_does(paths):

    assert paths.entry_file(12) == paths.entry_file("12")


def test_the_default_base_dir_is_the_chains_own_temp_folder():

    assert default_base_dir("tonality", "template_checklist") == os.path.join(
        "workflows", "tonality", ".temp", "template_checklist"
    )


def test_two_runs_of_one_chain_keep_separate_folders():
    base = default_base_dir("tonality", "template_checklist")

    assert RunPaths("run_0", base).run_dir() != RunPaths("run_1", base).run_dir()


@pytest.mark.parametrize("entry_id", ["case-1", "a.b", 12], ids=["plain", "dotted", "int"])
def test_an_id_read_off_its_own_file_name_is_the_id_that_named_it(paths, entry_id):
    filename = os.path.basename(paths.entry_file(entry_id))

    assert entry_id_from_filename(filename) == str(entry_id)


@pytest.mark.parametrize(
    "filename",
    ["notes.md", "case-1", "config.json", "case-1.txt.bak"],
    ids=["other-suffix", "no-suffix", "config", "backup"],
)
def test_a_file_this_loop_did_not_write_names_no_entry(filename):

    assert entry_id_from_filename(filename) is None


def test_slugify_tier_leaves_a_plain_name_untouched():
    assert slugify_tier("balanced") == "balanced"


def test_slugify_tier_replaces_a_colon_separated_model_id():
    assert slugify_tier("qwen3.5:0.8b") == "qwen3.5-0.8b"


def test_slugify_tier_collapses_a_run_of_unsafe_characters_into_one_dash():
    assert slugify_tier("a///b") == "a-b"


def test_slugify_tier_strips_leading_and_trailing_unsafe_characters():
    assert slugify_tier(":fast:") == "fast"
