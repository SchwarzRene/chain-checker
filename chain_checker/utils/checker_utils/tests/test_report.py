import json
import os

from chain_checker.baseclasses.corpus.corpus import Corpus
from chain_checker.baseclasses.corpus.entry import Entry
from chain_checker.baseclasses.corpus.input import Input
from chain_checker.baseclasses.corpus.label import Label
from chain_checker.baseclasses.corpus.output import ModelOutput
from chain_checker.utils.checker_utils import report as report_module
from chain_checker.utils.checker_utils.report import (
    existing_check_dirs,
    resumable_check_dir,
    write_report,
)


class _FakeModel:
    def __init__(
        self,
        prompt: str = "You are a helpful assistant.",
        has_prompt: bool = True,
        tier: str = "cheap",
    ) -> None:
        self._prompt = prompt
        self._has_prompt = has_prompt
        self._tier = tier

    def get_type(self) -> str:
        return "tonality"

    def get_chain_type(self) -> str:
        return "template_checklist"

    def get_tier(self) -> str:
        return self._tier

    def get_config(self) -> dict:
        return {"type": "tonality", "chain": "template_checklist", "tier": self._tier}

    def get_run_info(self) -> dict:
        return {"chain": "template_checklist", "tier": self._tier, "model": "test-model"}

    def get_system_prompt(self) -> str:
        if not self._has_prompt:
            raise AttributeError("module 'x' has no attribute 'SYSTEM_PROMPT'")
        return self._prompt


def _corpus_with_one_match() -> Corpus:
    corpus = Corpus()
    corpus.add_entry(
        Entry(
            "a",
            Input({"text": "hi"}),
            Label({"tonality": "formal"}),
            {},
            ModelOutput("", {"tonality": "formal"}),
        )
    )
    return corpus


# --------------------------------------------------------------------------
# _next_check_dir
# --------------------------------------------------------------------------


def test_the_first_check_dir_under_an_empty_base_is_check_0(tmp_path):
    assert report_module._next_check_dir(str(tmp_path)) == str(tmp_path / "check_0")


def test_the_next_check_dir_follows_the_highest_existing_number(tmp_path):
    (tmp_path / "check_0").mkdir()
    (tmp_path / "check_2").mkdir()

    assert report_module._next_check_dir(str(tmp_path)) == str(tmp_path / "check_3")


def test_entries_that_dont_match_the_check_prefix_pattern_are_ignored(tmp_path):
    (tmp_path / "check_0").mkdir()
    (tmp_path / "not_a_check_dir").mkdir()
    (tmp_path / "check_abc").mkdir()

    assert report_module._next_check_dir(str(tmp_path)) == str(tmp_path / "check_1")


def test_a_tier_is_appended_as_a_cosmetic_suffix(tmp_path):
    assert report_module._next_check_dir(str(tmp_path), "balanced") == str(
        tmp_path / "check_0_balanced"
    )


def test_an_unsafe_tier_is_slugified_in_the_suffix(tmp_path):
    assert report_module._next_check_dir(str(tmp_path), "qwen3.5:0.8b") == str(
        tmp_path / "check_0_qwen3.5-0.8b"
    )


def test_a_suffixed_existing_dir_still_counts_toward_the_next_number(tmp_path):
    (tmp_path / "check_0_balanced").mkdir()
    (tmp_path / "check_1_fast").mkdir()

    assert report_module._next_check_dir(str(tmp_path), "balanced") == str(
        tmp_path / "check_2_balanced"
    )


# --------------------------------------------------------------------------
# existing_check_dirs / resumable_check_dir
# --------------------------------------------------------------------------


def _check_path(*parts: str) -> str:
    # cwd-relative, like find_corpus_path() - see the write_report tests below.
    return os.path.join("workflows", "tonality", ".temp", "template_checklist", *parts)


def _make_check_dir(tmp_path, name: str, prompt: str | None = None, tier: str = "cheap") -> None:
    """A check dir as a finished run leaves it: the saved prompt.txt and
    config.json LoopCache compares a resumed run against."""
    check_dir = tmp_path / "workflows" / "tonality" / ".temp" / "template_checklist" / name
    check_dir.mkdir(parents=True)
    (check_dir / "prompt.txt").write_text(
        "You are a helpful assistant." if prompt is None else prompt
    )
    (check_dir / "config.json").write_text(
        json.dumps({"type": "tonality", "chain": "template_checklist", "tier": tier})
    )


def test_existing_check_dirs_is_empty_when_nothing_exists_yet(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert existing_check_dirs(_FakeModel()) == []


def test_existing_check_dirs_lists_every_dir_newest_first(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _make_check_dir(tmp_path, "check_0_cheap")
    _make_check_dir(tmp_path, "check_2_other-tier", tier="other-tier")

    assert existing_check_dirs(_FakeModel()) == [
        _check_path("check_2_other-tier"),
        _check_path("check_0_cheap"),
    ]


def test_resumable_check_dir_is_none_when_nothing_exists_yet(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert resumable_check_dir(_FakeModel()) is None


def test_resumable_check_dir_returns_the_newest_matching_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _make_check_dir(tmp_path, "check_0_cheap")
    _make_check_dir(tmp_path, "check_2_cheap")

    assert resumable_check_dir(_FakeModel()) == _check_path("check_2_cheap")


def test_resumable_check_dir_keeps_the_exact_existing_name_tier_suffix_and_all(
    monkeypatch, tmp_path
):
    # The dir's name says "some-other-tier" but its config.json says this
    # model's own tier - the name is cosmetic, so --continue must land in the
    # real directory rather than a freshly-guessed "check_0_cheap".
    monkeypatch.chdir(tmp_path)
    _make_check_dir(tmp_path, "check_0_some-other-tier")

    assert resumable_check_dir(_FakeModel()) == _check_path("check_0_some-other-tier")


def test_a_dir_made_with_another_tier_is_not_resumable(monkeypatch, tmp_path):
    # The reported bug: --chain-tier fast after a thinking run used to land
    # in the thinking run's dir, wipe its cached predictions and overwrite
    # its report. It must be passed over instead.
    monkeypatch.chdir(tmp_path)
    _make_check_dir(tmp_path, "check_0_thinking", tier="thinking")

    assert resumable_check_dir(_FakeModel(tier="fast")) is None


def test_a_dir_made_with_another_prompt_is_not_resumable(monkeypatch, tmp_path):
    # Same protection for --prompt-file (and for a SYSTEM_PROMPT edited in
    # the chain's source since): same tier, so the directory name matches,
    # but the cached predictions still describe a different prompt.
    monkeypatch.chdir(tmp_path)
    _make_check_dir(tmp_path, "check_0_cheap", prompt="an older prompt")

    assert resumable_check_dir(_FakeModel(prompt="a candidate prompt")) is None


def test_an_older_matching_dir_is_resumed_past_a_newer_mismatching_one(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _make_check_dir(tmp_path, "check_0_cheap")
    _make_check_dir(tmp_path, "check_1_thinking", tier="thinking")

    assert resumable_check_dir(_FakeModel()) == _check_path("check_0_cheap")


def test_a_dir_that_never_saved_a_prompt_or_config_is_not_resumable(monkeypatch, tmp_path):
    # Nothing on disk proves it was made with this setup, so it is left alone.
    monkeypatch.chdir(tmp_path)
    base_dir = tmp_path / "workflows" / "tonality" / ".temp" / "template_checklist"
    (base_dir / "check_0_cheap").mkdir(parents=True)

    assert resumable_check_dir(_FakeModel()) is None


def test_a_chain_without_a_rewritable_prompt_matches_on_an_empty_prompt(monkeypatch, tmp_path):
    # LoopCache saves "" for such a chain (see current_prompt), so a run made
    # by one stays resumable rather than being wiped every time.
    monkeypatch.chdir(tmp_path)
    _make_check_dir(tmp_path, "check_0_cheap", prompt="")

    assert resumable_check_dir(_FakeModel(has_prompt=False)) == _check_path("check_0_cheap")


# --------------------------------------------------------------------------
# write_report
# --------------------------------------------------------------------------


def test_write_report_creates_metrics_json_and_report_html_under_check_0(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    corpus = _corpus_with_one_match()

    report_path = write_report(_FakeModel(), corpus)

    check_dir = (
        tmp_path / "workflows" / "tonality" / ".temp" / "template_checklist" / "check_0_cheap"
    )
    # write_report()'s path is built from a relative base dir (cwd-relative,
    # like find_corpus_path()), not an absolute one.
    assert report_path == os.path.join(
        "workflows", "tonality", ".temp", "template_checklist", "check_0_cheap", "report.html"
    )
    assert (check_dir / "report.html").is_file()
    assert (check_dir / "metrics.json").is_file()


def test_write_report_includes_the_prompt_when_the_chain_has_one(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    corpus = _corpus_with_one_match()

    write_report(_FakeModel(prompt="Be nice."), corpus)

    metrics_json = (
        tmp_path
        / "workflows"
        / "tonality"
        / ".temp"
        / "template_checklist"
        / "check_0_cheap"
        / "metrics.json"
    )
    assert '"prompt": "Be nice."' in metrics_json.read_text()


def test_write_report_forwards_failed_ids_into_parsing_metrics(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    corpus = _corpus_with_one_match()

    write_report(_FakeModel(), corpus, failed_ids={"a"})

    metrics_json = (
        tmp_path
        / "workflows"
        / "tonality"
        / ".temp"
        / "template_checklist"
        / "check_0_cheap"
        / "metrics.json"
    )
    parsing_metrics = json.loads(metrics_json.read_text())["Parsing-Metrics"]
    assert parsing_metrics == {
        "total": 1,
        "parsed": 0,
        "failed": 1,
        "failure_rate": 1.0,
        "failed_ids": ["a"],
    }


def test_write_report_without_failed_ids_reports_everything_parsed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    corpus = _corpus_with_one_match()

    write_report(_FakeModel(), corpus)

    metrics_json = (
        tmp_path
        / "workflows"
        / "tonality"
        / ".temp"
        / "template_checklist"
        / "check_0_cheap"
        / "metrics.json"
    )
    parsing_metrics = json.loads(metrics_json.read_text())["Parsing-Metrics"]
    assert parsing_metrics["failed"] == 0
    assert parsing_metrics["parsed"] == 1


def test_write_report_omits_the_prompt_when_the_chain_has_none(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    corpus = _corpus_with_one_match()

    write_report(_FakeModel(has_prompt=False), corpus)

    metrics_json = (
        tmp_path
        / "workflows"
        / "tonality"
        / ".temp"
        / "template_checklist"
        / "check_0_cheap"
        / "metrics.json"
    )
    assert '"prompt"' not in metrics_json.read_text()
