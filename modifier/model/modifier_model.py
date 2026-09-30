import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from chain_checker.baseclasses.metrics.names import (
    accuracy_metrics,
    labeller_metrics,
    language_metrics,
    modification_metrics,
    modifier_token_usage_metrics,
    negative_predicted_metrics,
    parsing_metrics,
    text_length_metrics,
)
from chain_checker.modifier.llm.litellm import (
    DEFAULT_LITELLM_APP_LABEL,
    DEFAULT_LITELLM_TIER,
    LiteLLM,
)
from chain_checker.modifier.llm.llm_baseclass import LLM
from chain_checker.modifier.llm.ollama import Ollama
from chain_checker.modifier.model.compact_format import (
    format_accuracy_report,
    format_false_examples_report,
    format_parse_failure_report,
)
from chain_checker.modifier.model.extraction import extract_new_prompt
from chain_checker.modifier.model.instructions import (
    base_run_note,
    build_system_instructions,
    length_budget_note,
    measurement_note,
    stagnation_note,
    zero_false_examples_note,
)
from chain_checker.modifier.model.mismatches import simplify_mispredictions
from chain_checker.modifier.model.parsing_report import build_parse_failure_report
from chain_checker.modifier.model.persistence import save_json, save_text
from chain_checker.modifier.model.rule_trend import (
    RuleResults,
    best_rule_runs,
    format_rule_trend,
    rule_results_from_accuracy,
)
from chain_checker.utils.console import link_print
from chain_checker.utils.placeholders import detect_placeholders

_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")

# Runs in a row without a new best before the modifier is told to stop
# making small edits and restructure the prompt instead.
DEFAULT_STAGNATION_PATIENCE = 2
# A new best has to beat the old one by more than this to count as a gain:
# the target model is sampled, and unchanged rule text alone moves a 30-case
# run by about this much.
DEFAULT_STAGNATION_TOLERANCE = 0.02

# Runs at the end of the history that are always shown in full. Older runs
# are shown only if they are the best overall or the best for some rule.
DEFAULT_RECENT_RUNS_SHOWN = 2
# How much longer than the best prompt a rewrite may be before it is told to
# merge or delete wording instead of adding to it.
DEFAULT_LENGTH_GROWTH = 0.10

_EXCLUDED_FROM_MODIFIER = frozenset(
    (
        modification_metrics,
        text_length_metrics,
        language_metrics,
        labeller_metrics,
        modifier_token_usage_metrics,
    )
)


def _metric_results(data: list[dict[str, Any]], name: str) -> dict[str, Any]:
    # `or {}`, not `.get("results", {})`: a matching item with an explicit
    # "results": None must fall back too, the same way every other reader in
    # this package treats a present-but-None field as absent.
    return next((item.get("results") or {} for item in data if item.get("name") == name), {})


def _format_metric_definitions(descriptions: dict[str, str]) -> str:
    if not descriptions:
        return ""

    lines = [f"- {name}: {description}" for name, description in descriptions.items()]
    return "\n<-----------------METRIC-DEFINITIONS----------------->\n" + "\n".join(lines) + "\n"


def _format_run_block(
    index: int,
    prompt: str,
    accuracy_items: list[dict[str, Any]],
    negative_prompts: dict[str, dict[str, Any]],
    parse_failures: dict[str, dict[str, Any]],
    shown_in: Mapping[str, int],
) -> str:
    return (
        f"\n<--------------------Run-{index}--------------------->\n"
        f"\n<--------------------LLM-PROMPT--------------------->\n"
        f"{prompt}"
        f"\n<-----------------ACCURACY-REPORT------------------->\n"
        f"{format_accuracy_report(accuracy_items)}"
        f"\n<-----------------FALSE-EXAMPLES-REPORT------------->\n"
        f"{format_false_examples_report(negative_prompts, shown_in)}"
        f"\n<-----------------PARSE-FAILURE-REPORT--------------->\n"
        f"{format_parse_failure_report(parse_failures, shown_in)}"
    )


def _format_omitted_run(index: int, accuracy: float | None) -> str:
    score = "unscored" if accuracy is None else f"accuracy {accuracy:.3f}"
    return (
        f"\n<--------------------Run-{index}--------------------->\n"
        f"(omitted to save space - {score}; its prompt and reports are no longer shown)\n"
    )


@dataclass
class _RunRecord:
    """One completed rewrite attempt, replayed into the modifier's next
    instruction by `ModifierModel._build_instruction`. The report parts are
    kept raw and rendered per instruction: which runs are shown in full
    decides which run first prints a case's input."""

    prompt: str
    accuracy_items: list[dict[str, Any]]
    negative_prompts: dict[str, dict[str, Any]]
    parse_failures: dict[str, dict[str, Any]]
    accuracy: float | None = None
    rule_results: RuleResults = field(default_factory=dict)

    @property
    def had_mispredictions(self) -> bool:
        return bool(self.negative_prompts)


def _overall_accuracy(data: list[dict[str, Any]]) -> float | None:
    accuracy = _metric_results(data, accuracy_metrics).get("accuracy")
    # bool is an int subclass - a stray True must not read as 100%.
    if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)):
        return None
    return float(accuracy)


def _runs_since_last_gain(accuracies: list[float | None], tolerance: float) -> int:
    """How many runs have passed since the last one that beat every earlier
    run by more than `tolerance`. A run with no accuracy counts as no gain,
    and the first run with an accuracy always counts as a gain - it sets the
    reference the later runs are measured against."""
    last_gain = None
    reference = None
    for index, accuracy in enumerate(accuracies):
        if accuracy is None:
            continue
        if reference is None or accuracy > reference + tolerance:
            last_gain = index
        reference = accuracy if reference is None else max(reference, accuracy)
    return 0 if last_gain is None else len(accuracies) - 1 - last_gain


def _best_run_index(accuracies: list[float | None]) -> int | None:
    """The best-scoring run; the earliest wins a tie, since it is the same
    score reached with the prompt that had less time to accumulate patches."""
    scored = [(i, a) for i, a in enumerate(accuracies) if a is not None]
    return max(scored, key=lambda pair: (pair[1], -pair[0]))[0] if scored else None


def _case_count(rule_results: RuleResults) -> int:
    """How many cases one run scored each checked value on, read off the
    confusion counts; 0 when the run reported none (single-criterion chain)."""
    totals = (
        sum(results.get(key, 0) for key in ("TT", "TF", "FT", "FF"))
        for results in rule_results.values()
    )
    return max(totals, default=0)


class ModifierModel:
    """Rewrites a chain's system prompt each epoch by feeding an LLM the full
    history of every attempt so far - past prompts, their accuracy, and
    concrete mispredictions - and asking for one improved prompt back.

    `replay_run` alone reconstructs that history without calling the
    modifier LLM, which is what lets a `--continue`'d loop rebuild its past
    epochs before `modify` starts producing new rewrites again.
    """

    def __init__(
        self,
        llm: LLM | None = None,
        system_instructions: str | None = None,
        required_placeholders: tuple[str, ...] = (),
        stagnation_patience: int = DEFAULT_STAGNATION_PATIENCE,
        stagnation_tolerance: float = DEFAULT_STAGNATION_TOLERANCE,
        recent_runs_shown: int = DEFAULT_RECENT_RUNS_SHOWN,
        length_growth: float = DEFAULT_LENGTH_GROWTH,
    ) -> None:
        self._llm = llm if llm is not None else LLM()
        self._required_placeholders = tuple(required_placeholders)
        self._stagnation_patience = stagnation_patience
        self._stagnation_tolerance = stagnation_tolerance
        self._recent_runs_shown = recent_runs_shown
        self._length_growth = length_growth
        self._system_instructions = (
            system_instructions
            if system_instructions is not None
            else build_system_instructions(self._required_placeholders)
        )

        self._runs: list[_RunRecord] = []
        self._metric_descriptions: dict[str, str] = {}

        self._last_usage: dict[str, int] = {}
        self._cumulative_usage: dict[str, int] = dict.fromkeys(_USAGE_KEYS, 0)

    def init_ollama(self, model: str = "qwen3.5:4b") -> None:
        self._llm = Ollama(model)

    def init_litellm(
        self, app_label: str = DEFAULT_LITELLM_APP_LABEL, tier: str = DEFAULT_LITELLM_TIER
    ) -> None:
        self._llm = LiteLLM(app_label=app_label, tier=tier)

    def get_last_usage(self) -> dict[str, int]:
        return dict(self._last_usage)

    def get_cumulative_usage(self) -> dict[str, int]:
        return dict(self._cumulative_usage)

    def get_run_info(self) -> dict[str, str]:
        return self._llm.get_run_info()

    def _missing_placeholders(self, prompt: str) -> list[str]:
        try:
            present = detect_placeholders(prompt)
        except ValueError:
            # An unparsable prompt can't be checked for its placeholders
            # either - treat all of them as missing so the caller still
            # gets a warning instead of a silent pass.
            return list(self._required_placeholders)
        return [v for v in self._required_placeholders if v not in present]

    def replay_run(self, prompt: str, data: list[dict[str, Any]]) -> None:
        mispredicted = _metric_results(data, negative_predicted_metrics)
        parsing = _metric_results(data, parsing_metrics)

        accuracy_items = [
            item
            for item in data
            if item.get("name") != negative_predicted_metrics
            and item.get("name") not in _EXCLUDED_FROM_MODIFIER
        ]
        self._record_metric_descriptions(accuracy_items)

        negative_prompts = simplify_mispredictions(mispredicted)
        parse_failures = build_parse_failure_report(parsing, mispredicted)

        self._runs.append(
            _RunRecord(
                prompt=prompt,
                accuracy_items=accuracy_items,
                negative_prompts=negative_prompts,
                parse_failures=parse_failures,
                accuracy=_overall_accuracy(data),
                rule_results=rule_results_from_accuracy(_metric_results(data, accuracy_metrics)),
            )
        )

    def _record_metric_descriptions(self, accuracy_items: list[dict[str, Any]]) -> None:
        # Last non-empty wins, not first: a `--continue`'d run replays past
        # epochs with a placeholder description before any new epoch
        # supplies the real one - keeping the first value seen would freeze
        # that placeholder in place for the rest of the run.
        for item in accuracy_items:
            name, description = item.get("name"), item.get("description")
            if name and description:
                self._metric_descriptions[name] = description

    def _shown_runs(self) -> set[int]:
        """Runs printed in full: the latest few, plus any older run a rewrite
        may need as a base - the best overall and the best for each rule.
        Every other run is reduced to a one-line stub."""
        recent = range(max(0, len(self._runs) - self._recent_runs_shown), len(self._runs))
        shown = set(recent)
        best = _best_run_index([run.accuracy for run in self._runs])
        if best is not None:
            shown.add(best)
        shown |= best_rule_runs([run.rule_results for run in self._runs])
        return shown

    def _format_history(self) -> list[str]:
        shown_runs = self._shown_runs()
        # id -> the run whose report printed that case's input in full; every
        # later shown run only references it, so each text is sent once.
        shown_in: dict[str, int] = {}
        blocks = []
        for index, run in enumerate(self._runs):
            if index not in shown_runs:
                blocks.append(_format_omitted_run(index, run.accuracy))
                continue
            blocks.append(
                _format_run_block(
                    index,
                    run.prompt,
                    run.accuracy_items,
                    run.negative_prompts,
                    run.parse_failures,
                    dict(shown_in),
                )
            )
            for entry_id in (*run.negative_prompts, *run.parse_failures):
                shown_in.setdefault(entry_id, index)
        return blocks

    def _build_instruction(self) -> str:
        parts = [self._system_instructions, _format_metric_definitions(self._metric_descriptions)]
        parts.extend(self._format_history())
        parts.append(
            format_rule_trend([run.rule_results for run in self._runs], self._stagnation_tolerance)
        )
        parts.extend(self._guidance_notes())

        if not self._runs[-1].had_mispredictions:
            parts.append(zero_false_examples_note())
        else:
            parts.append(self._stagnation_note())

        parts.append("\n<-----------------NEW-LLM-PROMPT--------------------->\n")
        return "".join(parts)

    def _guidance_notes(self) -> list[str]:
        """Notes that steer the rewrite toward small, attributable steps:
        how coarse the measurement is, which run to build on, how long the
        result may be. Each is left out when its input is unknown."""
        notes = []
        case_count = _case_count(self._runs[-1].rule_results)
        if case_count:
            notes.append(measurement_note(case_count, self._stagnation_tolerance))

        best = _best_run_index([run.accuracy for run in self._runs])
        if best is None:
            return notes

        latest = len(self._runs) - 1
        if best != latest:
            notes.append(base_run_note(best, self._runs[best].accuracy, latest))
        notes.append(length_budget_note(len(self._runs[best].prompt), self._length_growth))
        return notes

    def _stagnation_note(self) -> str:
        """The note that tells the modifier to restructure rather than
        reword, or "" while the score is still moving."""
        accuracies = [run.accuracy for run in self._runs]
        runs_without_gain = _runs_since_last_gain(accuracies, self._stagnation_tolerance)
        if runs_without_gain < self._stagnation_patience:
            return ""

        best_run = _best_run_index(accuracies)
        best_accuracy = accuracies[best_run]
        link_print(
            f"(R)-(MODIFIER) No accuracy gain in {runs_without_gain} run(s) - asking the "
            f"modifier for a drastic rewrite (best so far: Run-{best_run} at {best_accuracy:.3f})."
        )
        return stagnation_note(runs_without_gain, best_run, best_accuracy)

    def modify(self, prompt: str, data: list[dict[str, Any]], save_dir: str) -> str:
        """Runs one epoch: records this attempt, asks the modifier LLM for a
        rewrite, and returns the new prompt. Also writes the instruction,
        raw output, token usage, and new prompt to `save_dir` as a side
        effect, so a run stays inspectable after the fact."""
        os.makedirs(save_dir, exist_ok=True)

        self.replay_run(prompt, data)

        instruction = self._build_instruction()
        save_text("modifier_instruction.txt", instruction, save_dir)

        raw_output = self._llm(instruction)
        save_text("modifier_output.txt", raw_output, save_dir)

        self._record_llm_usage(save_dir)

        new_prompt = extract_new_prompt(raw_output, fallback_prompt=prompt)
        self._warn_if_placeholders_frozen(new_prompt)

        save_text("modifier_new_prompt.txt", new_prompt, save_dir)
        return new_prompt

    def _record_llm_usage(self, save_dir: str) -> None:
        self._last_usage = self._llm.get_last_usage()
        for key in self._cumulative_usage:
            self._cumulative_usage[key] += self._last_usage.get(key, 0)
        save_json("modifier_token_usage.json", self._last_usage, save_dir)

    def _warn_if_placeholders_frozen(self, new_prompt: str) -> None:
        missing = self._missing_placeholders(new_prompt)
        if missing:
            link_print(
                f"(R)-(MODIFIER) note: the rewrite bakes in today's value(s) for "
                f"placeholder(s) {missing} directly instead of keeping them live - "
                f"valid (see instructions.py), but future runs won't auto-update "
                f"this content anymore unless a later rewrite reintroduces the "
                f"literal token."
            )
