from chain_checker.baseclasses.chain.model import Model as Chain
from chain_checker.utils.console import link_print
from chain_checker.utils.errors import fail


def apply_prompt_file(model: Chain, prompt_file: str, tag: str) -> None:
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
    except OSError as e:
        fail(f"Could not open prompt file '{prompt_file}'.\n  Reason: {e.strerror or e}")

    link_print(
        f"{tag} Using candidate prompt from '{prompt_file}' instead of "
        f"chain '{model.get_chain_type()}'s real SYSTEM_PROMPT"
    )
    model.set_new_system_prompt(prompt)


def print_nothing_to_continue(
    tag: str, existing_count: int, base_dir: str, fresh: str, mismatch: str
) -> None:
    """Says why --continue starts `fresh` anyway: nothing exists under
    `base_dir` yet, or every existing run fails the `mismatch` check."""
    if not existing_count:
        link_print(
            f"{tag} --continue given but no previous run exists under {base_dir} - "
            f"starting {fresh} fresh."
        )
        return

    link_print(
        f"{tag} --continue given, but none of the {existing_count} existing run(s) "
        f"under {base_dir} {mismatch}. Leaving them untouched and starting {fresh} "
        f"fresh instead."
    )
