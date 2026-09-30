import os

import pytest

from chain_checker.baseclasses.loop.center import ClassificationLoop
from chain_checker.utils.errors import CheckerError

from .conftest import FakeModel


@pytest.fixture
def make_loop(base_dir):
    def _make(corpus, model, run_id="modification_0") -> ClassificationLoop:
        return ClassificationLoop(run_id, corpus, model, base_dir)

    return _make


def _entry_file(loop, entry_id) -> str:
    return os.path.join(loop.get_run_dir(), "entries", f"{entry_id}.txt")


def test_every_entry_is_predicted_once(make_corpus, model, make_loop):
    make_loop(make_corpus("a", "b", "c"), model).loop()

    assert model.inputs == ["a", "b", "c"]


def test_each_prediction_is_set_on_its_own_entry(make_corpus, model, make_loop):
    corpus = make_corpus("a", "b")

    make_loop(corpus, model).loop()

    assert corpus.get_by_id("b").get_model_output().get_convo() == ("the conversation for b")


def test_each_prediction_is_saved_for_a_later_run(make_corpus, model, make_loop):
    loop = make_loop(make_corpus("a", "b"), model)

    loop.loop()

    assert sorted(os.listdir(os.path.dirname(_entry_file(loop, "a")))) == [
        "a.txt",
        "b.txt",
    ]


def test_the_prompt_and_config_the_run_used_are_saved(make_corpus, model, make_loop):

    loop = make_loop(make_corpus("a"), model)

    loop.loop()

    assert os.path.isfile(os.path.join(loop.get_run_dir(), "prompt.txt"))
    assert os.path.isfile(os.path.join(loop.get_run_dir(), "config.json"))


def test_a_progress_line_is_printed_for_every_entry_processed(
    make_corpus, model, make_loop, capsys
):

    make_loop(make_corpus("a", "b"), model).loop()

    printed = capsys.readouterr().out
    assert "(1/2) entry 'a'" in printed
    assert "(2/2) entry 'b'" in printed


def test_the_run_dir_is_the_one_the_cache_writes_to(make_corpus, model, make_loop):
    loop = make_loop(make_corpus("a"), model)

    loop.loop()

    assert os.path.isdir(loop.get_run_dir())


def test_an_empty_corpus_calls_nothing(make_corpus, model, make_loop):
    make_loop(make_corpus(), model).loop()

    assert model.inputs == []


def test_an_empty_corpus_says_so(make_corpus, model, make_loop, capsys):
    make_loop(make_corpus(), model).loop()

    assert "corpus is empty" in capsys.readouterr().out


def test_an_empty_corpus_writes_no_run_at_all(make_corpus, model, make_loop):

    loop = make_loop(make_corpus(), model)

    loop.loop()

    assert not os.path.exists(loop.get_run_dir())


def test_a_second_run_of_the_same_id_calls_the_model_for_nothing(make_corpus, model, make_loop):
    make_loop(make_corpus("a", "b"), model).loop()

    second_model = FakeModel()
    make_loop(make_corpus("a", "b"), second_model).loop()

    assert second_model.inputs == []


def test_a_second_run_still_grades_every_entry(make_corpus, model, make_loop):

    make_loop(make_corpus("a", "b"), model).loop()

    corpus = make_corpus("a", "b")
    make_loop(corpus, FakeModel()).loop()

    assert corpus.get_by_id("a").get_model_output().get_output() == {"passed": True}


def test_a_rewritten_prompt_re_runs_every_entry(make_corpus, model, make_loop):
    make_loop(make_corpus("a", "b"), model).loop()

    second_model = FakeModel(prompt="a rewritten prompt")
    make_loop(make_corpus("a", "b"), second_model).loop()

    assert second_model.inputs == ["a", "b"]


def test_a_recalled_entry_is_named_when_debugging(make_corpus, model, make_loop, capsys):
    make_loop(make_corpus("a"), model).loop()
    capsys.readouterr()

    make_loop(make_corpus("a"), FakeModel()).loop(debug=True)

    assert "recalled from disk, skipping" in capsys.readouterr().out


def test_nothing_per_entry_is_narrated_without_debug(make_corpus, model, make_loop, capsys):
    make_loop(make_corpus("a"), model).loop()

    assert "calling the model" not in capsys.readouterr().out


def test_debug_names_the_entry_it_is_calling(make_corpus, model, make_loop, capsys):
    make_loop(make_corpus("a"), model).loop(debug=True)

    assert "calling the model" in capsys.readouterr().out


def test_a_failing_entry_does_not_stop_the_ones_after_it(make_corpus, make_loop):
    model = FakeModel(fails=["b"])

    make_loop(make_corpus("a", "b", "c"), model).loop()

    assert model.inputs == ["a", "b", "c"]


def test_a_failing_entry_is_reported_with_its_input_and_the_reason(make_corpus, make_loop, capsys):
    make_loop(make_corpus("a"), FakeModel(fails=["a"])).loop()

    printed = capsys.readouterr().out
    assert "entry 'a'" in printed
    assert "could not parse the reply" in printed
    assert "continuing with the rest of the corpus" in printed


def test_a_failing_entry_keeps_the_empty_prediction_it_started_with(make_corpus, make_loop):

    corpus = make_corpus("a")

    make_loop(corpus, FakeModel(fails=["a"])).loop()

    assert corpus.get_by_id("a").get_model_output().get_output() == {}


def test_a_failing_entry_saves_nothing_so_a_later_run_retries_it(make_corpus, make_loop):
    loop = make_loop(make_corpus("a", "b"), FakeModel(fails=["a"]))

    loop.loop()

    assert not os.path.exists(_entry_file(loop, "a"))
    assert os.path.isfile(_entry_file(loop, "b"))


def test_a_failing_entry_is_still_timed(make_corpus, make_loop, capsys):

    make_loop(make_corpus("a", "b"), FakeModel(fails=["a"])).loop()

    assert "(1/2) entry 'a'" in capsys.readouterr().out


def test_the_ids_that_failed_are_reported(make_corpus, make_loop):
    loop = make_loop(make_corpus("a", "b", "c"), FakeModel(fails=["a", "c"]))

    loop.loop()

    assert loop.get_failed_ids() == {"a", "c"}


def test_a_numbered_id_is_reported_as_the_corpus_carries_it(make_corpus, make_loop):

    loop = make_loop(make_corpus(12), FakeModel(fails=["12"]))

    loop.loop()

    assert loop.get_failed_ids() == {12}


def test_the_failed_ids_are_a_copy(make_corpus, make_loop):
    loop = make_loop(make_corpus("a"), FakeModel(fails=["a"]))
    loop.loop()

    loop.get_failed_ids().add("never-failed")

    assert loop.get_failed_ids() == {"a"}


def test_the_failed_ids_describe_the_run_that_just_happened(make_corpus, make_loop):

    model = FakeModel(fails=["a"])
    loop = make_loop(make_corpus("a", "b"), model)
    loop.loop()

    model.set_fails()
    loop.loop()

    assert loop.get_failed_ids() == set()


def test_a_fatal_error_ends_the_run_instead_of_costing_one_entry(make_corpus, make_loop):

    class _Fatal(FakeModel):
        def __call__(self, inp) -> None:
            raise CheckerError("the chain could not be rebuilt")

    with pytest.raises(SystemExit):
        make_loop(make_corpus("a", "b"), _Fatal()).loop()


def test_a_run_where_nothing_worked_names_the_last_error(make_corpus, make_loop, capsys):
    make_loop(make_corpus("a", "b"), FakeModel(fails=["a", "b"])).loop()

    out = capsys.readouterr().out
    assert "every attempted entry" in out
    assert "last error: RuntimeError" in out


def test_a_run_where_one_entry_worked_makes_no_last_error_summary(make_corpus, make_loop, capsys):

    make_loop(make_corpus("a", "b"), FakeModel(fails=["a"])).loop()

    assert "every attempted entry" not in capsys.readouterr().out


def test_a_fully_recalled_run_makes_no_last_error_summary(make_corpus, model, make_loop, capsys):

    make_loop(make_corpus("a"), model).loop()
    capsys.readouterr()

    make_loop(make_corpus("a"), FakeModel()).loop()

    assert "every attempted entry" not in capsys.readouterr().out
