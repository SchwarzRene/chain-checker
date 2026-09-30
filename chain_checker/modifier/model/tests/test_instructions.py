from chain_checker.modifier.model.instructions import (
    _placeholder_note,
    base_run_note,
    build_system_instructions,
    length_budget_note,
    measurement_note,
    stagnation_note,
    zero_false_examples_note,
)


def test_no_placeholders_gets_a_note_saying_there_is_nothing_to_keep_live():
    note = _placeholder_note(())

    assert "no {placeholder} tokens" in note


def test_one_placeholder_is_named_by_its_brace_form():
    note = _placeholder_note(("tone",))

    assert "{tone}" in note


def test_several_placeholders_are_all_named():
    note = _placeholder_note(("tone", "text"))

    assert "{tone}" in note
    assert "{text}" in note


def test_build_system_instructions_embeds_the_placeholder_note():
    instructions = build_system_instructions(("tone",))

    assert "{tone}" in instructions


def test_build_system_instructions_states_the_mandatory_rules():
    # The model must never echo a marker itself - the one hard rule the
    # rest of the module (extraction.py) depends on holding.
    instructions = build_system_instructions(())

    assert "Never write any marker" in instructions


def test_build_system_instructions_states_the_task_as_a_whole_prompt_not_a_diff():
    # The modifier owns the whole text and must hand back all of it.
    instructions = build_system_instructions(())

    assert "TASK: write a NEW, BETTER system prompt" in instructions
    assert "you own the whole text" in instructions
    assert "ONLY the complete new prompt" in instructions


def test_build_system_instructions_asks_for_one_hypothesis_from_the_best_base():
    instructions = build_system_instructions(())

    assert "BASE:" in instructions
    assert "ONE HYPOTHESIS:" in instructions
    assert "PATTERNS, NOT CASES:" in instructions
    assert "LENGTH:" in instructions


def test_measurement_note_states_the_size_of_one_case_and_the_noise_level():
    note = measurement_note(30, 0.02)

    assert "30 cases" in note
    assert "0.033" in note
    assert "+-0.02" in note


def test_base_run_note_names_the_best_and_the_latest_run():
    note = base_run_note(best_run=1, best_accuracy=0.6762, latest_run=7)

    assert "Run-1 scored best (0.676)" in note
    assert "Run-7" in note


def test_length_budget_note_allows_the_given_growth_over_the_base():
    note = length_budget_note(1000, 0.10)

    assert "1000 characters" in note
    assert "1100 characters or fewer" in note


def test_zero_false_examples_note_flags_that_accuracy_is_already_perfect():
    note = zero_false_examples_note()

    assert "1.0" in note
    assert "COMPLETE FREEDOM" in note


def test_stagnation_note_names_the_best_run_and_its_accuracy():
    note = stagnation_note(3, best_run=1, best_accuracy=0.8142)

    assert "last 3 runs" in note
    assert "Run-1" in note
    assert "0.814" in note


def test_stagnation_note_asks_for_a_complete_rewrite_not_another_small_edit():
    note = stagnation_note(3, best_run=0, best_accuracy=0.5)

    assert "STAGNATION NOTE" in note
    assert "COMPLETE FREEDOM" in note
    assert "from scratch" in note


def test_stagnation_note_is_not_tied_to_multi_rule_chains():
    # Single-criterion chains get the same note, so it must not lean on
    # per-rule reports that only multi-rule chains produce.
    note = stagnation_note(3, best_run=0, best_accuracy=0.5)

    assert "RULE-TREND" not in note
    assert "rule" not in note.lower()


def test_build_system_instructions_points_to_the_stagnation_note():
    # The REWRITING section asks to keep what works; it has to hand over
    # to the stagnation note, or the two contradict each other.
    assert "STAGNATION NOTE" in build_system_instructions(())
