import json

from chain_checker.modifier.llm.llm_baseclass import LLM
from chain_checker.modifier.llm.ollama import Ollama
from chain_checker.modifier.model.modifier_model import ModifierModel
from chain_checker.modifier.model.tests.conftest import FakeLLM

# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_a_default_instance_gets_its_own_llm_not_a_shared_one():
    # llm: LLM = LLM() as a default argument would evaluate once and hand
    # every instance that omits `llm` the same object - so a() calling the
    # model would leave its usage visible on b() too.
    a = ModifierModel()
    b = ModifierModel()

    assert a._llm is not b._llm


def test_required_placeholders_are_kept_as_a_tuple_even_if_given_a_list():
    modifier = ModifierModel(llm=FakeLLM(), required_placeholders=["tone", "text"])

    assert modifier._required_placeholders == ("tone", "text")


def test_a_custom_system_instructions_string_is_used_verbatim(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm, system_instructions="REWRITE THIS PLEASE")

    modifier.replay_run("prompt v1", make_metric_data())

    assert modifier._build_instruction().startswith("REWRITE THIS PLEASE")


# --------------------------------------------------------------------------
# Backend switches
# --------------------------------------------------------------------------


def test_init_ollama_swaps_in_a_real_ollama_backend():
    modifier = ModifierModel()

    modifier.init_ollama("qwen3.5:4b")

    assert isinstance(modifier._llm, Ollama)
    assert modifier.get_run_info() == {"backend": "ollama", "model": "qwen3.5:4b"}


# --------------------------------------------------------------------------
# Usage accessors copy rather than alias internal state
# --------------------------------------------------------------------------


def test_get_last_usage_and_get_cumulative_usage_hand_back_copies(
    fake_llm, make_metric_data, tmp_path
):
    modifier = ModifierModel(llm=fake_llm)
    modifier.modify("prompt v1", make_metric_data(), str(tmp_path))

    last = modifier.get_last_usage()
    cumulative = modifier.get_cumulative_usage()
    last["prompt_tokens"] = -1
    cumulative["prompt_tokens"] = -1

    assert modifier.get_last_usage()["prompt_tokens"] != -1
    assert modifier.get_cumulative_usage()["prompt_tokens"] != -1


def test_cumulative_usage_starts_at_zero_not_empty():
    assert ModifierModel(llm=LLM()).get_cumulative_usage() == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def test_cumulative_usage_adds_up_across_several_modify_calls(fake_llm, make_metric_data, tmp_path):
    modifier = ModifierModel(llm=fake_llm)

    modifier.modify("prompt v1", make_metric_data(), str(tmp_path / "epoch0"))
    modifier.modify("prompt v2", make_metric_data(), str(tmp_path / "epoch1"))

    assert modifier.get_cumulative_usage() == {
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30,
    }


# --------------------------------------------------------------------------
# replay_run / _build_instruction: assembling the modifier's own history
# --------------------------------------------------------------------------


def test_replay_run_alone_never_calls_the_llm(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm)

    modifier.replay_run("prompt v1", make_metric_data())

    assert fake_llm.seen_prompts == []


def test_the_instruction_replays_every_run_prompt_in_order(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm)
    modifier.replay_run("prompt v1", make_metric_data())
    modifier.replay_run("prompt v2", make_metric_data())

    instruction = modifier._build_instruction()

    v1_at = instruction.index("prompt v1")
    run_1_at = instruction.index("Run-1")
    v2_at = instruction.index("prompt v2")
    assert v1_at < run_1_at < v2_at


def test_metric_descriptions_are_recorded_once_and_reused_across_runs(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm)
    modifier.replay_run("prompt v1", make_metric_data())
    modifier.replay_run("prompt v2", make_metric_data())

    instruction = modifier._build_instruction()

    assert instruction.count("share of cases the chain got right") == 1


def test_a_later_runs_description_replaces_an_earlier_placeholder_one(fake_llm):
    # A `--continue`'d run replays past epochs with a placeholder
    # description before any new epoch supplies the real one - the real
    # description must win, not get permanently shadowed by the placeholder.
    modifier = ModifierModel(llm=fake_llm)
    modifier.replay_run(
        "prompt v1",
        [{"name": "Accuracy-Metrics", "description": "(historical run)", "results": {}}],
    )
    modifier.replay_run(
        "prompt v2",
        [
            {
                "name": "Accuracy-Metrics",
                "description": "share of cases the chain got right",
                "results": {},
            }
        ],
    )

    instruction = modifier._build_instruction()

    assert "share of cases the chain got right" in instruction
    assert "(historical run)" not in instruction


def test_a_metric_the_modifier_excludes_never_reaches_the_instruction(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm)

    modifier.replay_run("prompt v1", make_metric_data(excluded_metric="TextLength-Metrics"))

    assert "TextLength-Metrics" not in modifier._build_instruction()


def test_the_modifiers_own_token_usage_metric_never_reaches_the_instruction(
    fake_llm, make_metric_data
):
    # Written back into metrics.json after every rewrite epoch and replayed
    # on `--continue` - it must not be mistaken for a real accuracy metric.
    modifier = ModifierModel(llm=fake_llm)

    modifier.replay_run(
        "prompt v1", make_metric_data(excluded_metric="Modifier-Token-Usage-Metrics")
    )

    assert "Modifier-Token-Usage-Metrics" not in modifier._build_instruction()


def test_a_run_with_mispredictions_gets_no_zero_false_examples_note(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm)

    modifier.replay_run(
        "prompt v1",
        make_metric_data(
            mispredicted={
                "case-1": {"input": {"a": 1}, "true": {"x": True}, "predicted": {"x": False}}
            }
        ),
    )

    assert "NOTE ON THE MOST RECENT RUN ABOVE" not in modifier._build_instruction()


def test_a_run_with_nothing_mispredicted_gets_the_zero_false_examples_note(
    fake_llm, make_metric_data
):
    modifier = ModifierModel(llm=fake_llm)

    modifier.replay_run("prompt v1", make_metric_data(mispredicted={}))

    assert "NOTE ON THE MOST RECENT RUN ABOVE" in modifier._build_instruction()


def test_only_the_most_recent_runs_mispredictions_decide_the_zero_note(fake_llm, make_metric_data):
    # Run 0 had mispredictions, run 1 (the most recent) had none - the note
    # is about "nothing left to gain right now", so only run 1 must count.
    modifier = ModifierModel(llm=fake_llm)
    modifier.replay_run(
        "prompt v1",
        make_metric_data(mispredicted={"case-1": {"input": {}, "true": {}, "predicted": {}}}),
    )
    modifier.replay_run("prompt v2", make_metric_data(mispredicted={}))

    assert "NOTE ON THE MOST RECENT RUN ABOVE" in modifier._build_instruction()


def test_calling_build_instruction_repeatedly_does_not_grow_the_history(fake_llm, make_metric_data):
    # Regression guard for the run-record cache: _build_instruction must be
    # a pure read of already-recorded runs, not something that appends.
    modifier = ModifierModel(llm=fake_llm)
    modifier.replay_run("prompt v1", make_metric_data())

    first = modifier._build_instruction()
    second = modifier._build_instruction()

    assert first == second
    assert len(modifier._runs) == 1


def test_a_metric_with_an_explicit_none_results_is_treated_as_empty(fake_llm):
    # A present "results": None key must fall back the same way a missing
    # "results" key does, not surface the None to simplify_mispredictions().
    modifier = ModifierModel(llm=fake_llm)

    modifier.replay_run(
        "prompt v1",
        [
            {"name": "Negative-Predicted-Metrics", "results": None},
            {"name": "Parsing-Metrics", "results": None},
        ],
    )

    assert modifier._runs[-1].had_mispredictions is False


# --------------------------------------------------------------------------
# _missing_placeholders
# --------------------------------------------------------------------------


def test_missing_placeholders_lists_only_the_required_ones_the_prompt_lacks(fake_llm):
    modifier = ModifierModel(llm=fake_llm, required_placeholders=("tone", "text"))

    assert modifier._missing_placeholders("Only {tone} appears here.") == ["text"]


def test_missing_placeholders_is_empty_when_the_prompt_keeps_every_token_live(fake_llm):
    modifier = ModifierModel(llm=fake_llm, required_placeholders=("tone", "text"))

    assert modifier._missing_placeholders("Uses {tone} and {text} both.") == []


def test_an_unparsable_prompt_reports_every_required_placeholder_as_missing(fake_llm):
    # detect_placeholders() raises ValueError on an unbalanced brace - the
    # prompt can't be checked, so the caller still needs a warning rather
    # than a silent, unchecked pass.
    modifier = ModifierModel(llm=fake_llm, required_placeholders=("tone",))

    assert modifier._missing_placeholders("A stray { brace") == ["tone"]


# --------------------------------------------------------------------------
# modify(): the full epoch - save files, call the llm, extract, warn
# --------------------------------------------------------------------------


def test_modify_returns_the_llms_rewritten_prompt(fake_llm, make_metric_data, tmp_path):
    modifier = ModifierModel(llm=fake_llm)

    result = modifier.modify("prompt v1", make_metric_data(), str(tmp_path))

    assert result == "rewritten prompt"


def test_modify_saves_the_instruction_output_and_new_prompt_to_disk(
    fake_llm, make_metric_data, tmp_path
):
    modifier = ModifierModel(llm=fake_llm)

    modifier.modify("prompt v1", make_metric_data(), str(tmp_path))

    assert (tmp_path / "modifier_instruction.txt").exists()
    assert (tmp_path / "modifier_output.txt").read_text() == "rewritten prompt"
    assert (tmp_path / "modifier_new_prompt.txt").read_text() == "rewritten prompt"
    usage = json.loads((tmp_path / "modifier_token_usage.json").read_text())
    assert usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_modify_creates_save_dir_if_it_does_not_exist_yet(fake_llm, make_metric_data, tmp_path):
    modifier = ModifierModel(llm=fake_llm)
    save_dir = tmp_path / "not-created-yet"

    modifier.modify("prompt v1", make_metric_data(), str(save_dir))

    assert save_dir.is_dir()


def test_modify_falls_back_to_the_current_prompt_if_the_reply_has_nothing_usable(
    make_metric_data, tmp_path
):
    modifier = ModifierModel(llm=FakeLLM(reply="   "))

    result = modifier.modify("prompt v1", make_metric_data(), str(tmp_path))

    assert result == "prompt v1"


def test_modify_warns_when_the_rewrite_freezes_a_required_placeholder(
    make_metric_data, tmp_path, capsys
):
    modifier = ModifierModel(
        llm=FakeLLM(reply="a prompt with no placeholders at all"),
        required_placeholders=("tone",),
    )

    modifier.modify("Use {tone}.", make_metric_data(), str(tmp_path))

    assert "placeholder(s) ['tone']" in capsys.readouterr().out


def test_modify_does_not_warn_when_the_rewrite_keeps_its_placeholders_live(
    make_metric_data, tmp_path, capsys
):
    modifier = ModifierModel(
        llm=FakeLLM(reply="Still uses {tone} here."),
        required_placeholders=("tone",),
    )

    modifier.modify("Use {tone}.", make_metric_data(), str(tmp_path))

    assert capsys.readouterr().out == ""


def test_modify_tolerates_an_epoch_with_no_scoreable_metrics_at_all(fake_llm, tmp_path):
    modifier = ModifierModel(llm=fake_llm)

    result = modifier.modify("prompt v1", [], str(tmp_path))

    assert result == "rewritten prompt"


# --------------------------------------------------------------------------
# Stagnation - drastic rewrite once the score stops moving
# --------------------------------------------------------------------------

_ONE_MISS = {"case-1": {"input": {"a": 1}, "true": {"x": True}, "predicted": {"x": False}}}


def _replay_accuracies(modifier, make_metric_data, accuracies):
    for i, accuracy in enumerate(accuracies):
        modifier.replay_run(
            f"prompt v{i}", make_metric_data(accuracy=accuracy, mispredicted=_ONE_MISS)
        )


def test_no_stagnation_note_while_the_score_still_improves(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm)
    _replay_accuracies(modifier, make_metric_data, [0.60, 0.65, 0.70, 0.75])

    assert "STAGNATION NOTE:" not in modifier._build_instruction()


def test_no_stagnation_note_before_patience_runs_have_passed(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm)
    _replay_accuracies(modifier, make_metric_data, [0.80, 0.80])

    assert "STAGNATION NOTE:" not in modifier._build_instruction()


def test_two_flat_runs_trigger_the_stagnation_note_by_default(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm)
    _replay_accuracies(modifier, make_metric_data, [0.80, 0.80, 0.80])

    assert "STAGNATION NOTE: the last 2 runs" in modifier._build_instruction()


def test_three_flat_runs_after_the_best_trigger_the_stagnation_note(
    fake_llm, make_metric_data, capsys
):
    modifier = ModifierModel(llm=fake_llm)
    _replay_accuracies(modifier, make_metric_data, [0.70, 0.81, 0.80, 0.79, 0.81])

    instruction = modifier._build_instruction()

    assert "STAGNATION NOTE: the last 3 runs" in instruction
    # The first of two tied bests is named - it is the one the others failed to beat.
    assert "Run-1" in instruction
    assert "drastic rewrite" in capsys.readouterr().out


def test_gains_within_the_noise_tolerance_do_not_reset_the_count(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm, stagnation_tolerance=0.02)
    _replay_accuracies(modifier, make_metric_data, [0.800, 0.810, 0.815, 0.819])

    assert "STAGNATION NOTE: the last 3 runs" in modifier._build_instruction()


def test_a_real_gain_resets_the_count(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm)
    _replay_accuracies(modifier, make_metric_data, [0.80, 0.80, 0.80, 0.80, 0.90])

    assert "STAGNATION NOTE:" not in modifier._build_instruction()


def test_stagnation_patience_is_configurable(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm, stagnation_patience=1)
    _replay_accuracies(modifier, make_metric_data, [0.80, 0.80])

    assert "STAGNATION NOTE: the last 1 runs" in modifier._build_instruction()


def test_runs_without_an_accuracy_count_as_no_gain(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm)
    modifier.replay_run("prompt v0", make_metric_data(accuracy=0.8, mispredicted=_ONE_MISS))
    for i in range(1, 4):
        modifier.replay_run(f"prompt v{i}", make_metric_data(accuracy=None, mispredicted=_ONE_MISS))

    assert "STAGNATION NOTE: the last 3 runs" in modifier._build_instruction()


def test_no_stagnation_note_when_no_run_has_an_accuracy(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm)
    for i in range(4):
        modifier.replay_run(f"prompt v{i}", make_metric_data(accuracy=None, mispredicted=_ONE_MISS))

    assert "STAGNATION NOTE:" not in modifier._build_instruction()


def test_a_perfect_run_gets_the_zero_false_examples_note_not_the_stagnation_note(
    fake_llm, make_metric_data
):
    modifier = ModifierModel(llm=fake_llm)
    for i in range(4):
        modifier.replay_run(f"prompt v{i}", make_metric_data(accuracy=1.0, mispredicted={}))

    instruction = modifier._build_instruction()

    assert "STAGNATION NOTE:" not in instruction
    assert "NOTE ON THE MOST RECENT RUN ABOVE" in instruction


# --------------------------------------------------------------------------
# History size and cross-run evidence
# --------------------------------------------------------------------------

_LONG_TEXT = "ein langer Text " * 20
_LONG_MISS = {
    "case-1": {
        "input": {"text": _LONG_TEXT},
        "true": {"verdicts": {"hc": False}},
        "predicted": {"verdicts": {"hc": True}},
    }
}


def test_a_case_text_is_sent_once_however_many_runs_mispredict_it(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm)
    for i in range(3):
        modifier.replay_run(f"prompt v{i}", make_metric_data(mispredicted=_LONG_MISS))

    instruction = modifier._build_instruction()

    assert instruction.count(_LONG_TEXT) == 1
    assert instruction.count("(input: see Run-0)") == 2


def test_the_instruction_carries_a_rule_trend_across_runs(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm)
    for i, accuracy in enumerate((0.7, 0.6, 0.5)):
        data = make_metric_data(mispredicted=_LONG_MISS)
        data[0]["results"] = {
            "accuracy": accuracy,
            "rules": {
                "verdicts": {
                    "sub_rules": {
                        "hc": {
                            "results": {
                                "accuracy": accuracy,
                                "TT": 5,
                                "TF": 1,
                                "FT": 3 + i,
                                "FF": 5,
                            }
                        }
                    }
                }
            },
        }
        modifier.replay_run(f"prompt v{i}", data)

    instruction = modifier._build_instruction()

    assert "<-----------------RULE-TREND" in instruction
    assert "verdicts.hc | 0.700 1/3 | 0.600 1/4 | 0.500 1/5 | Run-0 |" in instruction
    assert "FALLING 2 runs in a row" in instruction


# --------------------------------------------------------------------------
# History pruning and guidance notes
# --------------------------------------------------------------------------


def _scored(make_metric_data, accuracy, *, cases=30, mispredicted=None):
    data = make_metric_data(accuracy=accuracy, mispredicted=mispredicted)
    data[0]["results"] = {
        "accuracy": accuracy,
        "rules": {
            "verdicts": {
                "sub_rules": {
                    "hc": {
                        "results": {
                            "accuracy": accuracy,
                            "TT": cases,
                            "TF": 0,
                            "FT": 0,
                            "FF": 0,
                        }
                    }
                }
            }
        },
    }
    return data


def test_old_runs_that_are_neither_recent_nor_best_shrink_to_a_stub(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm, recent_runs_shown=2)
    # Run-1 is the best; Runs 0 and 2 are old and worse; 3 and 4 are recent.
    for i, accuracy in enumerate((0.5, 0.9, 0.4, 0.6, 0.7)):
        modifier.replay_run(f"prompt v{i}", _scored(make_metric_data, accuracy))

    instruction = modifier._build_instruction()

    assert "prompt v0" not in instruction
    assert "prompt v2" not in instruction
    assert "Run-0" in instruction and "(omitted to save space - accuracy 0.500" in instruction
    for kept in ("prompt v1", "prompt v3", "prompt v4"):
        assert kept in instruction


def test_a_case_input_is_printed_again_once_its_first_run_is_omitted(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm, recent_runs_shown=1)
    miss = {"case-1": {"input": {"text": "UNIQUE-CASE-TEXT"}, "matches": {"verdicts": False}}}
    modifier.replay_run("prompt v0", _scored(make_metric_data, 0.5, mispredicted=miss))
    modifier.replay_run("prompt v1", _scored(make_metric_data, 0.9))
    modifier.replay_run("prompt v2", _scored(make_metric_data, 0.6, mispredicted=miss))

    instruction = modifier._build_instruction()

    assert "prompt v0" not in instruction
    assert "(input: see Run-0)" not in instruction


def test_the_base_note_points_at_the_best_run_when_the_latest_is_worse(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm)
    modifier.replay_run("prompt v0", _scored(make_metric_data, 0.8))
    modifier.replay_run("prompt v1", _scored(make_metric_data, 0.6))

    assert "BASE NOTE: Run-0 scored best (0.800)" in modifier._build_instruction()


def test_no_base_note_when_the_latest_run_is_the_best(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm)
    modifier.replay_run("prompt v0", _scored(make_metric_data, 0.6))
    modifier.replay_run("prompt v1", _scored(make_metric_data, 0.8))

    assert "BASE NOTE: Run-" not in modifier._build_instruction()


def test_a_tie_for_best_goes_to_the_earliest_run(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm)
    for i, accuracy in enumerate((0.8, 0.8, 0.5)):
        modifier.replay_run(f"prompt v{i}", _scored(make_metric_data, accuracy))

    assert "BASE NOTE: Run-0" in modifier._build_instruction()


def test_the_length_budget_is_relative_to_the_best_runs_prompt(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm, length_growth=0.5)
    modifier.replay_run("x" * 100, _scored(make_metric_data, 0.8))
    modifier.replay_run("y" * 400, _scored(make_metric_data, 0.6))

    instruction = modifier._build_instruction()

    assert "the base prompt is 100 characters" in instruction
    assert "150 characters or fewer" in instruction


def test_the_measurement_note_uses_the_case_count_of_the_latest_run(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm)
    modifier.replay_run("prompt v0", _scored(make_metric_data, 0.8, cases=40))

    assert "scored on 40 cases" in modifier._build_instruction()


def test_a_chain_without_per_rule_counts_gets_no_measurement_note(fake_llm, make_metric_data):
    modifier = ModifierModel(llm=fake_llm)
    modifier.replay_run("prompt v0", make_metric_data(accuracy=0.8))

    instruction = modifier._build_instruction()

    assert "MEASUREMENT NOTE: every checked value" not in instruction
    assert "LENGTH BUDGET" in instruction
