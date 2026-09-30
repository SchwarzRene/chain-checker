import json
from typing import Any

from chain_checker.baseclasses.corpus.output import ModelOutput

# Deliberately distinctive so a split() on them is safe; the parser assumes
# the raw conversation transcript before them never happens to contain this
# exact text.
PREDICTION_MARKER = "\n\n----- PREDICTION (json) -----\n"
TOKEN_USAGE_MARKER = "\n\n----- TOKEN USAGE (json) -----\n"


def render_prediction(predicted: ModelOutput) -> str:
    parts = [
        predicted.get_convo(),
        PREDICTION_MARKER,
        json.dumps(predicted.get_output(), indent=2, default=str),
    ]

    usage = predicted.get_token_usage()
    if usage:
        parts += [
            TOKEN_USAGE_MARKER,
            json.dumps(usage, indent=2, sort_keys=True),
        ]

    return "".join(parts)


def parse_prediction(content: str, path: str) -> ModelOutput:
    if PREDICTION_MARKER not in content:
        raise ValueError(f"missing prediction marker in '{path}'")

    conversation, rest = content.split(PREDICTION_MARKER, 1)
    raw_output, raw_usage = _split_token_usage(rest)

    return ModelOutput(
        conversation,
        _parse_object(raw_output, path, "prediction"),
        {} if raw_usage is None else _parse_object(raw_usage, path, "token usage"),
    )


def _split_token_usage(rest: str) -> tuple[str, str | None]:
    # render_prediction only writes this section when there is usage to
    # report (see its `if usage:` check), so a file without one is valid,
    # not corrupt.
    if TOKEN_USAGE_MARKER not in rest:
        return rest, None

    raw_output, raw_usage = rest.split(TOKEN_USAGE_MARKER, 1)
    return raw_output, raw_usage


def _parse_object(raw: str, path: str, section: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"could not parse the saved {section} JSON in '{path}': {e}") from e

    if not isinstance(value, dict):
        raise ValueError(
            f"the saved {section} in '{path}' is a {type(value).__name__}, not a json object"
        )
    return value
