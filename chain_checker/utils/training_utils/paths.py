import os

from chain_checker.baseclasses.loop.paths import numbered_dirs, slugify_tier
from chain_checker.utils.metrics_file import read_metrics_file

_EPOCH_PREFIX = "modification_"


def existing_run_numbers(base_dir: str) -> list[int]:
    return sorted(numbered_dirs(base_dir, "run"))


def run_dir(base_dir: str, run_number: int, tier: str | None = None) -> str:
    existing_name = numbered_dirs(base_dir, "run").get(run_number)
    if existing_name is not None:
        return os.path.join(base_dir, existing_name)

    suffix = f"_{slugify_tier(tier)}" if tier else ""
    return os.path.join(base_dir, f"run_{run_number}{suffix}")


def epoch_dir(run_dir: str, epoch: int) -> str:
    return os.path.join(run_dir, f"{_EPOCH_PREFIX}{epoch}")


def epoch_number(epoch_name: str) -> int:
    return int(epoch_name.removeprefix(_EPOCH_PREFIX))


def epoch_names(run_dir: str) -> list[str]:
    if not os.path.isdir(run_dir):
        return []
    return sorted(
        (name for name in os.listdir(run_dir) if name.startswith(_EPOCH_PREFIX)),
        key=epoch_number,
    )


def final_epoch_done(run_dir: str, epoch: int) -> bool:
    return read_metrics_file(os.path.join(epoch_dir(run_dir, epoch), "metrics.json")) is not None
