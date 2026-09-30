import json
import os
from typing import Any


def save_text(filename: str, content: str, save_dir: str) -> None:
    # Explicit encoding: content is LLM output, which can carry non-ASCII
    # text regardless of the host's default locale.
    with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as f:
        f.write(content)


def save_json(filename: str, data: Any, save_dir: str) -> None:
    try:
        text = json.dumps(data, indent=2, sort_keys=True)
    except TypeError:
        # sort_keys=True raises on a dict with mixed int/str keys (e.g.
        # entry ids) - still write the data, just unsorted.
        text = json.dumps(data, indent=2)
    with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as f:
        f.write(text)
