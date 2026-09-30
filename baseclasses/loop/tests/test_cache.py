import json
import os

import pytest

from chain_checker.baseclasses.corpus.output import ModelOutput
from chain_checker.baseclasses.loop.cache import LoopCache

from .conftest import NO_PROMPT, FakeModel


@pytest.fixture
def make_cache(base_dir):
    def _make(corpus, model, run_id="modification_0") -> LoopCache:
        return LoopCache(run_id, corpus, model, base_dir)

    return _make


def _prediction(entry_id) -> ModelOutput:
    return ModelOutput(f"the conversation for {entry_id}", {"passed": True}, {"total_tokens": 5})


def _save_run(cache, corpus) -> None:

    cache.save_prompt()
    cache.save_config()
    for entry in corpus:
        cache.save_prediction(entry.get_id(), _prediction(entry.get_id()))


def _prompt_path(cache) -> str:
    return os.path.join(cache.get_run_dir(), "prompt.txt")


def _config_path(cache) -> str:
    return os.path.join(cache.get_run_dir(), "config.json")


def test_a_run_with_no_base_dir_given_lands_in_the_chains_temp_folder(make_corpus, model):

    cache = LoopCache("modification_0", make_corpus("a"), model)

    assert cache.get_run_dir() == os.path.join(
        "workflows", "tonality", ".temp", "template_checklist", "modification_0"
    )


def test_the_files_of_a_run_go_where_the_paths_say(make_corpus, model, make_cache):
    cache = make_cache(make_corpus("a"), model)

    cache.save_prediction("a", _prediction("a"))

    assert os.path.isfile(cache.get_entry_path("a"))
    assert cache.get_entries_dir() == os.path.dirname(cache.get_entry_path("a"))


def test_the_prompt_is_saved_as_a_model_sees_it(make_corpus, model, make_cache):

    model.prompt = 'Rewrite {text}. Reply as {"ok": true}'
    cache = make_cache(make_corpus("a"), model)

    cache.save_prompt()

    with open(_prompt_path(cache)) as f:
        assert f.read() == 'Rewrite {text}. Reply as {"ok": true}'


def test_the_config_the_run_ran_under_is_saved_beside_it(make_corpus, model, make_cache):
    cache = make_cache(make_corpus("a"), model)

    cache.save_config()

    with open(_config_path(cache)) as f:
        assert json.load(f) == model.get_config()


def test_a_chain_with_no_prompt_of_its_own_saves_an_empty_one(make_corpus, make_cache):

    cache = make_cache(make_corpus("a"), FakeModel(prompt=NO_PROMPT))

    cache.save_prompt()

    with open(_prompt_path(cache)) as f:
        assert f.read() == ""


def test_a_prompt_that_failed_for_any_other_reason_is_not_saved_as_empty(make_corpus, make_cache):

    model = FakeModel(prompt_error=ValueError("the chain module is broken"))
    cache = make_cache(make_corpus("a"), model)

    with pytest.raises(ValueError):
        cache.save_prompt()


def test_a_prediction_is_saved_under_its_own_entry_id(make_corpus, model, make_cache):
    cache = make_cache(make_corpus("a", "b"), model)

    cache.save_prediction("b", _prediction("b"))

    assert os.listdir(cache.get_entries_dir()) == ["b.txt"]


def test_a_failure_is_saved_beside_the_cache_rather_than_in_it(make_corpus, model, make_cache):
    cache = make_cache(make_corpus("a"), model)

    cache.save_failure("a", ModelOutput("what the failed call sent and got", {}, {}))

    assert os.path.isfile(cache.get_failure_path("a"))
    assert not os.path.exists(cache.get_entries_dir())


def test_a_saved_failure_is_never_recalled_as_a_prediction(make_corpus, model, make_cache):
    # An empty output under entries/ would be served as a real answer and the
    # entry would never be retried - hence the separate directory.
    corpus = make_corpus("a", "b")
    cache = make_cache(corpus, model)
    cache.save_prompt()
    cache.save_config()
    cache.save_prediction("a", _prediction("a"))
    cache.save_failure("b", ModelOutput("what the failed call sent and got", {}, {}))

    assert cache.recall() == {"a"}


def test_nothing_is_recalled_from_a_folder_that_holds_no_run(make_corpus, model, make_cache):
    assert make_cache(make_corpus("a", "b"), model).recall() == set()


def test_a_matching_run_is_recalled_by_entry_id(make_corpus, model, make_cache):
    corpus = make_corpus("a", "b")
    cache = make_cache(corpus, model)
    _save_run(cache, corpus)

    assert cache.recall() == {"a", "b"}


def test_a_recalled_prediction_is_set_on_the_corpus_entry(make_corpus, model, make_cache):

    corpus = make_corpus("a")
    cache = make_cache(corpus, model)
    _save_run(cache, corpus)

    cache.recall()

    predicted = corpus.get_by_id("a").get_model_output()
    assert predicted.get_output() == {"passed": True}
    assert predicted.get_convo() == "the conversation for a"


def test_a_recalled_prediction_keeps_what_the_call_cost(make_corpus, model, make_cache):

    corpus = make_corpus("a")
    cache = make_cache(corpus, model)
    _save_run(cache, corpus)

    cache.recall()

    assert corpus.get_by_id("a").get_model_output().get_token_usage() == {"total_tokens": 5}


def test_a_numbered_entry_recalls_under_the_text_of_its_id(make_corpus, model, make_cache):
    corpus = make_corpus(12)
    cache = make_cache(corpus, model)
    _save_run(cache, corpus)

    assert cache.recall() == {"12"}


def test_a_matching_run_that_saved_no_entries_recalls_nothing(make_corpus, model, make_cache):
    cache = make_cache(make_corpus("a"), model)
    cache.save_prompt()
    cache.save_config()

    assert cache.recall() == set()


def test_a_file_for_an_entry_this_corpus_does_not_have_is_ignored(make_corpus, model, make_cache):

    corpus = make_corpus("a")
    cache = make_cache(corpus, model)
    _save_run(cache, corpus)
    cache.save_prediction("gone", _prediction("gone"))

    assert cache.recall() == {"a"}


def test_a_file_this_loop_did_not_write_is_ignored(make_corpus, model, make_cache):
    corpus = make_corpus("a")
    cache = make_cache(corpus, model)
    _save_run(cache, corpus)
    with open(os.path.join(cache.get_entries_dir(), "notes.md"), "w") as f:
        f.write("a note somebody left here")

    assert cache.recall() == {"a"}


def test_a_run_saved_under_a_different_prompt_is_not_recalled(make_corpus, model, make_cache):

    corpus = make_corpus("a")
    cache = make_cache(corpus, model)
    _save_run(cache, corpus)

    model.prompt = "a rewritten prompt"

    assert cache.recall() == set()


def test_a_run_saved_under_a_different_tier_is_not_recalled(make_corpus, model, make_cache):
    corpus = make_corpus("a")
    cache = make_cache(corpus, model)
    _save_run(cache, corpus)

    model.set_tier("thinking")

    assert cache.recall() == set()


def test_a_run_whose_config_cannot_be_read_is_not_recalled(make_corpus, model, make_cache):
    corpus = make_corpus("a")
    cache = make_cache(corpus, model)
    _save_run(cache, corpus)
    with open(_config_path(cache), "w") as f:
        f.write("{not json")

    assert cache.recall() == set()


@pytest.mark.parametrize("missing", ["prompt.txt", "config.json"])
def test_a_run_missing_either_half_of_its_identity_is_not_recalled(
    make_corpus, model, make_cache, missing
):
    corpus = make_corpus("a")
    cache = make_cache(corpus, model)
    _save_run(cache, corpus)
    os.remove(os.path.join(cache.get_run_dir(), missing))

    assert cache.recall() == set()


def test_a_stale_runs_predictions_are_discarded_rather_than_left_lying(
    make_corpus, model, make_cache
):

    corpus = make_corpus("a", "b")
    cache = make_cache(corpus, model)
    _save_run(cache, corpus)

    model.prompt = "a rewritten prompt"
    cache.recall()

    assert os.listdir(cache.get_entries_dir()) == []


def test_discarding_a_stale_run_says_so(make_corpus, model, make_cache, capsys):
    corpus = make_corpus("a")
    cache = make_cache(corpus, model)
    _save_run(cache, corpus)

    model.prompt = "a rewritten prompt"
    cache.recall()

    assert "different prompt or chain config" in capsys.readouterr().out


def test_a_file_the_loop_did_not_write_survives_a_discard(make_corpus, model, make_cache):

    corpus = make_corpus("a")
    cache = make_cache(corpus, model)
    _save_run(cache, corpus)
    with open(os.path.join(cache.get_entries_dir(), "notes.md"), "w") as f:
        f.write("a note somebody left here")

    model.prompt = "a rewritten prompt"
    cache.recall()

    assert os.listdir(cache.get_entries_dir()) == ["notes.md"]


def test_a_fresh_run_discards_nothing_and_says_nothing(make_corpus, model, make_cache, capsys):
    make_cache(make_corpus("a"), model).recall()

    assert capsys.readouterr().out == ""


def test_an_entry_file_that_cannot_be_parsed_costs_only_that_entry(make_corpus, model, make_cache):
    corpus = make_corpus("a", "b")
    cache = make_cache(corpus, model)
    _save_run(cache, corpus)
    with open(cache.get_entry_path("a"), "w") as f:
        f.write("half a file, killed mid-save")

    assert cache.recall() == {"b"}


def test_an_entry_file_that_cannot_be_parsed_says_which_and_why(
    make_corpus, model, make_cache, capsys
):
    corpus = make_corpus("a")
    cache = make_cache(corpus, model)
    _save_run(cache, corpus)
    with open(cache.get_entry_path("a"), "w") as f:
        f.write("half a file, killed mid-save")

    cache.recall()

    printed = capsys.readouterr().out
    assert "could not recall" in printed
    assert "a.txt" in printed


def test_an_entry_whose_file_could_not_be_read_keeps_its_empty_prediction(
    make_corpus, model, make_cache
):

    corpus = make_corpus("a")
    cache = make_cache(corpus, model)
    _save_run(cache, corpus)
    with open(cache.get_entry_path("a"), "w") as f:
        f.write("half a file, killed mid-save")

    cache.recall()

    assert corpus.get_by_id("a").get_model_output().get_output() == {}
