import os

from chain_checker.baseclasses.chain.model import Model
from chain_checker.baseclasses.loop.cache import saved_config_matches
from chain_checker.modifier.model import ModifierModel
from chain_checker.utils.console import link_print
from chain_checker.utils.training_utils.metrics_io import (
    load_previous_metrics,
    reconstruct_metrics_list,
)
from chain_checker.utils.training_utils.paths import (
    epoch_dir,
    epoch_names,
    existing_run_numbers,
    run_dir,
)


def resumable_run_number(base_dir: str, model: Model) -> int | None:
    """The newest run under `base_dir` that `model` may continue, or None
    when there is no such run and a fresh one has to be started.

    A run is continuable when its first epoch was made with the same chain
    config (type/chain/tier) as `model` - otherwise --continue would append
    this tier's epochs to another tier's run, leaving one run_N whose epochs
    were trained against two different models and one summary averaging over
    both. Unlike the checker's equivalent, only the config is compared, never
    the prompt: rewriting the prompt each epoch is what a training run *is*,
    so a prompt that has moved on is the normal case, not a mismatch.

    A run with no epochs on disk yet counts as continuable - it is an empty
    shell left by an interrupted start (write_summary runs before epoch 0),
    with no foreign results in it to protect.
    """
    config = model.get_config()
    for number in reversed(existing_run_numbers(base_dir)):
        candidate = run_dir(base_dir, number)
        if not epoch_names(candidate) or saved_config_matches(epoch_dir(candidate, 0), config):
            return number
    return None


def resume_epoch(model: Model, run_dir: str) -> int:
    epoch = 0
    while os.path.isfile(os.path.join(epoch_dir(run_dir, epoch), "modifier_new_prompt.txt")):
        epoch += 1

    if epoch == 0:
        return 0

    own_prompt_path = os.path.join(epoch_dir(run_dir, epoch), "prompt.txt")
    resume_from = (
        own_prompt_path
        if os.path.isfile(own_prompt_path)
        else os.path.join(epoch_dir(run_dir, epoch - 1), "modifier_new_prompt.txt")
    )

    with open(resume_from, "r", encoding="utf-8") as f:
        model.set_new_system_prompt(f.read())

    link_print(
        f"(R)-(MODIFIER) Continuing {run_dir} - epoch(s) 0-{epoch - 1} already "
        f"finished, restoring the prompt from {resume_from} and continuing "
        f"from epoch {epoch}."
    )
    return epoch


def replay_modifier_history(modifier: ModifierModel, run_dir: str, upto_epoch: int) -> None:
    for e in range(upto_epoch):
        current_epoch_dir = epoch_dir(run_dir, e)

        prompt_path = os.path.join(current_epoch_dir, "prompt.txt")
        if not os.path.isfile(prompt_path):
            continue
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read()

        run_report_data = load_previous_metrics(current_epoch_dir)
        if not run_report_data:
            continue

        modifier.replay_run(prompt, reconstruct_metrics_list(run_report_data))
