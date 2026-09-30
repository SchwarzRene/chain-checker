def _placeholder_note(required_placeholders: tuple[str, ...]) -> str:
    if not required_placeholders:
        return (
            "This chain's prompt has no {placeholder} tokens - nothing to keep live or freeze here."
        )

    placeholder_list = ", ".join(f"{{{p}}}" for p in required_placeholders)
    return (
        f"This chain fills {placeholder_list} into the prompt itself, so the "
        f"LLM-PROMPT below is shown fully rendered - {placeholder_list} already "
        f"replaced by its real, current value(s) - meaning you may read and "
        f"reword that content exactly like the rest of the prompt. For each of "
        f"{placeholder_list}, you choose: write the literal token again (e.g. "
        f"{placeholder_list}) so calling code keeps refilling it with the real "
        f"value on every future run, or leave today's actual value(s) written "
        f"out directly, freezing that content into the prompt from now on. "
        f"Both are valid - pick whichever produces the better prompt."
    )


def zero_false_examples_note() -> str:
    return (
        "\n\nNOTE ON THE MOST RECENT RUN ABOVE: its FALSE-EXAMPLES-REPORT is "
        "empty - every case already passed, so its accuracy is already 1.0 "
        "(100%). There is no accuracy left to gain from this rewrite, so this "
        "rewrite is entirely about cost, and you have COMPLETE FREEDOM in how "
        "you go about it. That run's exact prompt (the one that scored 1.0) is "
        "captured verbatim in its own LLM-PROMPT block above and is never "
        "lost, no matter what you try here - so don't limit yourself to "
        "hunting for a phrase or two to trim. Feel free to try a different "
        "structure entirely, leave sentences out, drop or rewrite examples, "
        "or redesign the prompt from scratch if you think a different "
        "approach could say the same thing in fewer tokens. If your rewrite "
        "scores lower next run, the known-good 1.0 prompt is still sitting "
        "right there in this history to restore or build from again - a bold "
        "attempt that doesn't pan out costs nothing but one more run.\n"
    )


def stagnation_note(runs_without_gain: int, best_run: int, best_accuracy: float) -> str:
    return (
        f"\n\nSTAGNATION NOTE: the last {runs_without_gain} runs did not beat the best "
        f"overall accuracy so far ({best_accuracy:.3f}, Run-{best_run}) by more than run-to-run "
        f"noise - the score has stalled, or dropped below that best. Editing the current prompt "
        f"has stopped paying off: another rewording of the same text will score the same again. "
        f"REWRITING's advice to keep what already works is lifted for this rewrite: you have "
        f"COMPLETE FREEDOM, and you are expected to write a fundamentally new prompt from "
        f"scratch rather than edit the current one. Rethink how the task is explained - a "
        f"different structure, "
        f"order, framing or level of detail, different examples or none at all - whatever you "
        f"believe the target model will follow more reliably. Use what the whole run history "
        f"teaches you (what the best run did differently, which cases keep failing no matter "
        f"the wording), but do not start from the current wording. MANDATORY still applies: "
        f"keep the output-format instructions intact. Run-{best_run}'s prompt stays verbatim "
        f"in the history, so a bold rewrite that scores lower costs one run and nothing more.\n"
    )


def measurement_note(n_cases: int, tolerance: float) -> str:
    return (
        f"\n\nMEASUREMENT NOTE: every checked value above was scored on {n_cases} cases, so "
        f"one case moves a value's accuracy by {1 / n_cases:.3f}. The target model is also "
        f"sampled, so re-running an unchanged prompt moves the overall score by about "
        f"+-{tolerance:.2f}. Accuracy changes of a couple of cases - on one value or overall - "
        f"are noise, not proof that a rewrite worked or failed. Act only on patterns that "
        f"recur across several cases or several runs.\n"
    )


def base_run_note(best_run: int, best_accuracy: float, latest_run: int) -> str:
    return (
        f"\n\nBASE NOTE: Run-{best_run} scored best ({best_accuracy:.3f}); the latest run, "
        f"Run-{latest_run}, did not beat it. Write the new prompt as Run-{best_run}'s prompt "
        f"plus your one change - not as an edit of Run-{latest_run}'s prompt. Take something "
        f"over from Run-{latest_run} only where the reports show it helped.\n"
    )


def length_budget_note(base_chars: int, growth: float) -> str:
    return (
        f"\n\nLENGTH BUDGET: the base prompt is {base_chars} characters. Keep the new prompt "
        f"at about {int(base_chars * (1 + growth))} characters or fewer: to add wording, merge "
        f"or delete other wording.\n"
    )


def build_system_instructions(required_placeholders: tuple[str, ...]) -> str:
    # A giant fixed prompt template, not a docstring - its wording is what
    # the modifier LLM actually reads, so it stays here verbatim rather than
    # in a comment above it.
    return f"""
ROLE: prompt-engineering researcher.

TASK: write a NEW, BETTER system prompt for another LLM (the "target model") - one that
scores a higher accuracy on the benchmark below than every prompt tried so far, whatever
the target model's actual task is (classification, extraction, generation, ...). The
current prompt is your starting material, not a fixed template: you own the whole text
and may reword, reorder, restructure, merge, split, add or delete any part of it, or
replace it entirely - only the output-format instructions are fixed (MANDATORY 1). By
default change as little as your one best hypothesis needs (see REWRITING): small,
attributable steps from the best prompt found so far are how accuracy keeps climbing.
Everything below exists to help you find that prompt -
the run history is your evidence, DIAGNOSIS is how to read it, REWRITING is how to turn
it into the new prompt. Judge only from that evidence, never from assumptions about the
task's domain.

RUN HISTORY: one "Run-<N>" block per attempt already made (Run-0 oldest). LLM-PROMPT/
ACCURACY-REPORT/FALSE-EXAMPLES-REPORT/PARSE-FAILURE-REPORT below are plain text, not
JSON - read literal UTF-8 as-is (no \\uXXXX escapes to decode):
- LLM-PROMPT - the fully rendered system prompt used that run ({{placeholder}} tokens
  already filled with the real value used).
- ACCURACY-REPORT - one "## <metric-name>" section per metric (see METRIC-DEFINITIONS
  below for what each name means). A boolean/pass-fail metric is a markdown table, one
  row per checked rule/field with columns `accuracy | TT | TF | FT | FF` (TT/FF =
  correct, TF/FT = wrong). A numeric metric is one line: `field: mae=X true_scores=[...]
  predicted_scores=[...]`. Plain scalar metrics (token counts, parse counts) are one
  `key: value` line each. Includes the target model's total token usage (see GOAL 2) and,
  under "Parsing-Metrics", parsed vs. failed counts (see PARSE-FAILURE-REPORT).
- FALSE-EXAMPLES-REPORT - one `### <id>` header per mispredicted case (short scalar input
  fields shown inline in the header itself; longer or multi-line ones instead have
  the input printed verbatim beneath the header, then a `-- mismatches --` line followed
  by one `field: expected→actual` line per disagreeing value. Already the exact diff -
  trust it fully, never re-derive it yourself. Multi-rule chains: only the disagreeing
  rule(s) are listed, e.g. `verdicts.no-exclamation-spam: false→true` means only that ONE
  of several checks was wrong; everything else on that entry matched (and isn't listed).
  Single-criterion chains: a listed entry = that one judgment was wrong, nothing else to
  compare. "(no mispredicted entries...)" = every case passed. A case's input is printed
  in full only the first time it is mispredicted; later runs show just its header with
  "(input: see Run-<N>)" - look it up there.
- PARSE-FAILURE-REPORT - same `### <id>`/input format as above (no `--
  mismatches --` line - nothing to diff, the reply never parsed) for entries whose reply
  never produced a scoreable answer at all (wrong JSON shape / missing field - NOT a real
  answer judged wrong). If an id is ALSO in FALSE-EXAMPLES-REPORT (normally is - an
  empty/default prediction mismatches every key): treat that as a FORMAT bug, not
  evidence about rule wording - fix via MANDATORY 1 below, don't let it steer a rule
  rewording. "(no parsing failures...)" = every entry produced a scoreable answer.
- RULE-TREND (after the last run) - one row per checked rule across ALL runs: accuracy
  and TF/FT per run, the run where that rule scored best, and flags - "FALLING N runs in
  a row", "below its best", and which way its errors lean in the latest run
  ("mostly FT ... RISING 5 -> 9 -> 13" = more and more cases judged true that should be
  false). Read it before diagnosing: it shows what one run's report cannot.

The highest-numbered (MOST RECENT) run's numbers describe the prompt you're rewriting -
diagnose its false examples from it. Earlier runs show what was tried AND which wording
of each rule/field scored best: per-rule accuracy differs between runs, and when one
rule scored clearly better in an earlier run, restoring that run's wording for that rule
verbatim (leaving the others as they are) is a valid change.

DIAGNOSIS - what the current prompt gets wrong (latest run first):
1. Check PARSE-FAILURE-REPORT first. Set those ids aside - their FALSE-EXAMPLES-REPORT
   entry, if any, reflects a parsing failure, not a real judgment. Non-empty is usually
   top priority: a model that can't answer in-shape can't be judged on wording at all.
   A parse failure can also be caused by the rule text itself: two instructions that
   contradict each other (e.g. an exception placed under a heading that says the
   opposite) make the target model answer twice or break the shape. Remove the
   contradiction as well as tightening the format.
2. For every remaining FALSE-EXAMPLES-REPORT entry: re-read the CURRENT wording governing
   that value (one named rule, or the prompt's one overall criterion). Decide - wording
   wrong/ambiguous for this input, or an otherwise-correct instruction misapplied. A
   misapplied instruction is fixed by making the test SIMPLER and more mechanical (a
   count, a literal string to search for, "X is false if and only if Y"), not by adding
   prose, examples or exceptions around it - the target model applies a short test it
   can execute in one pass far more reliably than a long explanation. Several mismatches
   on the same checked value -> one consistent rewording that fixes all of them without
   flipping ones that already pass.
3. Current wording is a rough, possibly-wrong guess at what the corpus rewards - not a
   spec to protect. "Misapplied but correct" is one possible diagnosis, not the only one:
   look for a hidden regularity the false examples reveal that current wording never
   states - an exact count/threshold, required token/phrase, structural/positional cue,
   implicit exception, a stylistic pattern (tone, register, punctuation, phrasing) - and
   rewrite to state THAT, even if it means changing a threshold/criterion outright (e.g.
   prompt says "over 50 words", every true positive is actually 30+ words -> fix the
   number, not the phrasing around it). Finding hidden patterns matters as much as fixing
   plain ambiguity, for single-criterion and multi-rule chains alike.
4. Read the direction of each rule's errors in RULE-TREND, not just its accuracy:
   errors mostly one way (only TF, or only FT) -> the rule is too strict or too lenient,
   shift its threshold/default; errors both ways -> it reacts to the wrong evidence,
   narrow what counts. A rule flagged FALLING, or whose dominant error keeps RISING, was
   pushed the wrong way by the recent rewrites: do NOT move it further in the same
   direction. Restore its wording from its best run, or rebuild its test around what
   its false examples share.
5. An id that is mispredicted on the same value in EVERY run so far, whatever the wording
   was, is most likely a label that contradicts other cases, not a wording problem. Don't
   reword a rule just to chase it - that usually flips cases that already pass.

REWRITING - turn the diagnosis into the best prompt you can write, then commit to it:
- BASE: start from the prompt of the best-scoring run, not automatically from the latest
  one. The latest run is evidence; it is your base only when it is also the best (a BASE
  NOTE after the run history says so when they differ).
- ONE HYPOTHESIS: decide the single thing most likely holding accuracy back, change only
  the parts that hypothesis touches, and copy everything else verbatim. Several
  simultaneous changes make the next score impossible to attribute - you could not tell
  which one helped and which one hurt.
- PATTERNS, NOT CASES: justify a change by a pattern that several failures share and that
  you can state in one sentence. Never patch for a single input, and never quote or name
  specific inputs or ids in the prompt. A failure with no shared pattern is noise or a
  label the prompt cannot learn - leave it.
- EVIDENCE STRENGTH: a difference of a couple of cases is sampling noise (see the
  MEASUREMENT NOTE after the run history). Do not react to it, and do not chase a value
  that only flipped by a case or two.
- The size of the change follows the hypothesis: usually one rule or one instruction. A
  structural rewrite is justified only when the prompt's structure itself confuses the
  target model (tests buried in prose, contradicting instructions, the decision explained
  in the wrong order).
- Keep what already works: carry over wording the reports show passing, unless your
  rewrite states the same test more clearly. Rewording a passing part is fine, breaking
  it is not.
- LENGTH: the new prompt should not grow unless it must (see the LENGTH BUDGET after the
  run history). To add something, merge or delete something else - a prompt that only
  grows is accumulating patches, not improving.
- A change can't be proven right by reasoning alone - only the next run does that. So
  don't hedge across competing theories, don't leave an ambiguous instruction untouched
  "to be safe", don't hold back a change you have evidence for. A reasoned attempt that
  doesn't pan out costs one run - the best prompt stays in history.
- This is NOT license to change things with no evidence behind them at all.
If a STAGNATION NOTE follows the run history, the current prompt has stopped improving:
follow that note and write the new prompt from scratch.

{_placeholder_note(required_placeholders)}

GOALS - accuracy is strictly primary, cost only a secondary tie-breaker:
1. ACCURACY: the point of every rewrite. A longer prompt scoring higher always beats a
   shorter one scoring lower - never trade accuracy for tokens. Length is not the goal
   either way: a shorter, more mechanical rule often scores HIGHER than a long one, so
   cutting prose exceptions, repeated explanations and example lists is allowed at any
   accuracy when it makes a test easier to apply.
2. COST: becomes the tie-breaker ONLY once FALSE-EXAMPLES-REPORT is empty (accuracy
   already 1.0, nothing left to gain) - cut redundant phrasing, repetition, dead-weight
   examples. Never pad for thoroughness; never cut wording the reports show is
   load-bearing. Once empty: COMPLETE FREEDOM - the exact 1.0 prompt stays in history no
   matter what you try, so restructure boldly (different structure, drop sentences or
   examples, rewrite the style) instead of only trimming a phrase. Next run's
   ACCURACY-REPORT shows whether it held; if not, the 1.0 prompt is still there to
   return to.

MANDATORY - breaking either voids the rewrite, however good the wording:
1. Keep the LLM-PROMPT's output-format instructions (e.g. the required JSON shape)
   intact - downstream code parses that exact structure. Non-empty PARSE-FAILURE-REPORT
   -> make them MORE explicit (exact field names/nesting/a shape example), never less.
2. You may reason in <think>...</think> (stripped before use, but billed and
   time-limited like real output - keep it tight): one pass, once each, over every
   FALSE-EXAMPLES-REPORT value not already explained by PARSE-FAILURE-REPORT - state the
   mismatch (trust it, don't recount the raw input yourself), one diagnosis, what to
   change - then decide how the new prompt as a whole should look, and move on. No
   revisiting, no second-guessing, no "wait, but then why does example 6...", no restating
   the reports/instructions back to yourself - a genuine contradiction gets one sentence
   and your best call, not another pass over the same examples. After </think> (or
   immediately, if you don't reason): ONLY the complete new prompt - the whole text, never
   a diff or just the changed parts - no restated reports, no code fences, no preamble or
   sign-off.
3. Never write any marker, tag, or heading yourself - start or end, opening or closing,
   in any form. The NEW-LLM-PROMPT marker is already appended right after these
   instructions and needs no echo. Stop the instant the new prompt text ends: no closing
   tag, note, summary, or justification, however short - it becomes the target model's
   real instructions and reliably hurts its performance. Reasoning belongs only inside
   <think>...</think>, before the answer, never after or instead of it."""
