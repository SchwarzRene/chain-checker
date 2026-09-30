from typing import Any


def diff_mismatches(
    true_data: dict[str, Any], predicted_data: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Field-by-field diff between a case's true and predicted output.

    A one-level-nested field (true_data[key] is itself a dict) is flattened
    to "key.subkey" per disagreeing subkey, matching how the compact report
    tables key a multi-rule chain's per-rule verdicts.
    """
    diff: dict[str, dict[str, Any]] = {}
    for key, true_value in true_data.items():
        pred_value = predicted_data.get(key)
        if isinstance(true_value, dict):
            pred_sub = pred_value if isinstance(pred_value, dict) else {}
            for subkey, sub_true in true_value.items():
                sub_pred = pred_sub.get(subkey)
                if sub_pred != sub_true:
                    diff[f"{key}.{subkey}"] = {"true": sub_true, "predicted": sub_pred}
        elif pred_value != true_value:
            diff[key] = {"true": true_value, "predicted": pred_value}
    return diff


def simplify_mispredictions(entries: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        entry_id: {
            "input": entry.get("input", {}),
            "mismatches": diff_mismatches(entry.get("true") or {}, entry.get("predicted") or {}),
        }
        for entry_id, entry in entries.items()
    }
