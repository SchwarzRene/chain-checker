import os
import re

_ENTRY_SUFFIX = ".txt"

_UNSAFE_TIER_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def default_base_dir(app_label: str, chain_type: str) -> str:

    return os.path.join("workflows", app_label, ".temp", chain_type)


def numbered_dirs(base_dir: str, prefix: str) -> dict[int, str]:
    """Maps N to the dir name for every "<prefix>_N" or tier-suffixed
    "<prefix>_N_<tier>" under `base_dir` - the suffix never affects numbering."""
    if not os.path.isdir(base_dir):
        return {}

    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)(?:_.*)?$")
    dirs: dict[int, str] = {}
    for name in os.listdir(base_dir):
        match = pattern.match(name)
        if match:
            dirs[int(match.group(1))] = name
    return dirs


def slugify_tier(tier: str) -> str:
    """A tier name with anything unsafe for a directory name (e.g. the ':' in
    an Ollama model id) replaced by '-'."""
    return _UNSAFE_TIER_CHARS.sub("-", tier).strip("-")


def entry_id_from_filename(filename: str) -> str | None:

    entry_id, ext = os.path.splitext(filename)
    return entry_id if ext == _ENTRY_SUFFIX else None


class RunPaths:
    def __init__(self, run_id: str, base_dir: str) -> None:
        self._run_id = run_id
        self._base_dir = base_dir

    @classmethod
    def for_run_dir(cls, run_dir: str) -> "RunPaths":
        base_dir, run_id = os.path.split(run_dir)
        return cls(run_id, base_dir)

    def run_dir(self) -> str:
        return os.path.join(self._base_dir, self._run_id)

    def entries_dir(self) -> str:
        return os.path.join(self.run_dir(), "entries")

    def failures_dir(self) -> str:
        return os.path.join(self.run_dir(), "failures")

    def prompt_file(self) -> str:
        return os.path.join(self.run_dir(), "prompt.txt")

    def config_file(self) -> str:
        return os.path.join(self.run_dir(), "config.json")

    def entry_file(self, entry_id: str | int) -> str:
        return os.path.join(self.entries_dir(), f"{entry_id}{_ENTRY_SUFFIX}")

    def failure_file(self, entry_id: str | int) -> str:
        return os.path.join(self.failures_dir(), f"{entry_id}{_ENTRY_SUFFIX}")
