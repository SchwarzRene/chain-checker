from collections import Counter
from collections.abc import Iterator
from typing import Any

from chain_checker.baseclasses.corpus.entry import Entry
from chain_checker.baseclasses.corpus.label import (
    Kind,
    kind_conflict_message,
    value_kind,
)
from chain_checker.baseclasses.metrics.m_base import BaseMetrics
from chain_checker.baseclasses.metrics.m_boolean import BooleanMetrics
from chain_checker.baseclasses.metrics.m_float import FloatMetrics
from chain_checker.baseclasses.metrics.m_value import ValueMetrics
from chain_checker.utils.errors import fail

_HEADLINE_KEY = "accuracy"
_RULES = "rules"
_RESULTS = "results"
_SUB_RULES = "sub_rules"


# Same sentinel convention as m_mispredicted's _MISSING: a key the
# prediction never carried must read as distinct from an expected value of
# None, which is legal here too.
_MISSING = "<missing>"


RulePath = tuple[str, ...]


def iter_rule_results(results: dict[str, Any]) -> Iterator[tuple[RulePath, dict[str, Any]]]:
    # Mirrors compute()'s own output shape - {key: {"results": ...}} for a
    # plain rule, {key: {"sub_rules": {subkey: {"results": ...}}}} for one
    # expanded from a dict-of-bool key - so a caller (e.g. the report) can
    # walk every leaf metric without caring which shape a given rule took.
    for key, node in results.get(_RULES, {}).items():
        if _RESULTS in node:
            yield (key,), node[_RESULTS]
            continue
        for subkey, sub_node in node.get(_SUB_RULES, {}).items():
            yield (key, subkey), sub_node[_RESULTS]


def rule_label(path: RulePath) -> str:
    return " / ".join(path)


class SingleValueMetrics(BaseMetrics):
    def __init__(self, name: str) -> None:
        description = (
            "For each rule in the label, compares the true and predicted value. "
            "Every rule is reported under 'rules', keyed by its corpus key, as "
            "either {'results': ...} or - for a dict-of-bool key expanded into one "
            "sub-metric per sub-key - {'sub_rules': {subkey: {'results': ...}}}. "
            "Boolean rules report 'accuracy' plus a confusion matrix: "
            "TT = correctly predicted true, "
            "TF = should have been true but the prediction did not agree, "
            "FT = should have been false but the prediction did not agree, "
            "FF = correctly predicted false. "
            "Numeric rules report 'mae' (the average absolute error over the pairs "
            "that could be compared, with 'scored'/'unscored' saying how much of the "
            "corpus that was) plus an 'accuracy' counting how often the prediction "
            "agreed within tolerance, and the raw 'true_scores'/'predicted_scores'. "
            "Any other rule - a string, a list, a mapping that is not all bools - "
            "reports 'accuracy' with 'matched'/'mismatched' counts. "
            "Agreement everywhere is decided by the same comparison a graded "
            "PASS/FAIL uses, so this metric and that verdict cannot disagree. "
            "The top-level 'accuracy' key is the fraction of all individual rule "
            "checks that agreed, weighted by how many entries each rule was "
            "actually checked on, "
            "e.g. {'rules': {'rule1': {'results': {'accuracy': 0.8, 'TT': 4, "
            "'TF': 1, 'FT': 0, 'FF': 5}}, "
            "'verdicts': {'sub_rules': {'r1': {'results': {'accuracy': 1.0, ...}}}}}, "
            "'accuracy': 0.83}."
        )
        super().__init__(name, description)
        self._data: dict[RulePath, BaseMetrics] = {}
        self._counts: Counter[RulePath] = Counter()
        self._kinds: dict[str, tuple[Kind, str | int, Any]] = {}

    def add_entry(self, entry: Entry) -> None:
        true = entry.get_output()
        pred = entry.get_model_output()
        pred_data = pred.get()

        for key in true.get_keys():
            true_value = true.get_value(key)
            pred_value = pred_data.get(key, _MISSING)

            kind = value_kind(true_value)
            self._check_kind(entry, key, kind, true_value)

            if kind == "bools":
                self._add_dict_entry(
                    key,
                    true_value,
                    pred_value if isinstance(pred_value, dict) else {},
                )
                continue

            self._record((key,), key, kind, true_value, pred_value)

    def _record(
        self, path: RulePath, name: str, kind: Kind, true_value: Any, pred_value: Any
    ) -> None:
        if path not in self._data:
            self._data[path] = self._metric_for(name, kind)

        self._data[path].add(true_value, pred_value)
        self._counts[path] += 1

    def _check_kind(self, entry: Entry, key: str, kind: Kind, true_value: Any) -> None:
        if key not in self._kinds:
            self._kinds[key] = (kind, entry.get_id(), true_value)

        established, first_id, first_value = self._kinds[key]
        if established != kind:
            fail(kind_conflict_message(key, first_id, first_value, entry.get_id(), true_value))

    def _add_dict_entry(
        self, key: str, true_dict: dict[str, Any], pred_dict: dict[str, Any]
    ) -> None:
        # An unrecognized key in the prediction (one the label never checks)
        # is treated as the whole dict being unproduced, not as a partial
        # answer scored only on the keys that do match - a shape mismatch
        # like this is more likely a chain output bug than a genuine partial
        # prediction, and scoring the recognized subset would hide it.
        if not pred_dict.keys() <= true_dict.keys():
            pred_dict = {}

        for subkey, true_subvalue in true_dict.items():
            self._record(
                (key, subkey),
                subkey,
                "bool",
                true_subvalue,
                pred_dict.get(subkey, _MISSING),
            )

    @staticmethod
    def _metric_for(name: str, kind: Kind) -> BaseMetrics:
        if kind == "bool":
            return BooleanMetrics(
                name=name,
                description=f"Boolean rule '{name}': accuracy and a TT/TF/FT/FF matrix.",
            )

        if kind == "number":
            return FloatMetrics(
                name=name,
                description=f"Numeric rule '{name}': mean absolute error and accuracy.",
            )

        return ValueMetrics(
            name=name,
            description=f"Rule '{name}', neither bool nor number: agreement only.",
        )

    def _calculate_mean_accuracy(self, results: dict[RulePath, dict[str, Any]]) -> float:
        # Weighted by how many entries each rule was actually checked on
        # (self._counts), not by 1 / number_of_rules - a rule scored on every
        # entry should outweigh one that only fired for a rare sub-case.
        # Exclude numeric rules (those with 'mae') from accuracy averaging - they
        # are tracked separately in the MAE chart and shouldn't contaminate the
        # boolean/categorical accuracy metric.
        checks = 0
        agreed = 0.0
        for path, result in results.items():
            if "mae" in result:
                continue
            weight = self._counts[path]
            checks += weight
            agreed += result["accuracy"] * weight

        return agreed / checks if checks else 0.0

    def compute(self) -> dict[str, Any]:
        results = {path: metric.compute() for path, metric in self._data.items()}

        rules: dict[str, Any] = {}
        for path, result in results.items():
            if len(path) == 1:
                rules[path[0]] = {_RESULTS: result}
                continue

            key, subkey = path
            group = rules.setdefault(key, {_SUB_RULES: {}})
            group[_SUB_RULES][subkey] = {_RESULTS: result}

        return {
            _RULES: rules,
            _HEADLINE_KEY: self._calculate_mean_accuracy(results),
        }
