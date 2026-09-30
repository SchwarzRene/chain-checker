# ruff: noqa: E402
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import chain_checker.modifier.parser as parser
from chain_checker.baseclasses.chain.model import Model
from chain_checker.baseclasses.corpus import Corpus
from chain_checker.baseclasses.loop.paths import default_base_dir
from chain_checker.modifier.model import ModifierModel
from chain_checker.utils.bootstrap import setup_django, use_generous_llm_timeout
from chain_checker.utils.checker_utils.startup import (
    apply_prompt_file,
    print_nothing_to_continue,
)
from chain_checker.utils.checker_utils.validation import find_corpus_path, validate_corpus
from chain_checker.utils.console import animate_chain, link_print
from chain_checker.utils.errors import fail
from chain_checker.utils.training_utils import paths, reports, resume
from chain_checker.utils.training_utils.epoch import run_epoch


def _new_run_dir(base_dir: str, model: Model, existing_runs: list[int]) -> str:
    run_number = (existing_runs[-1] + 1) if existing_runs else 0
    return paths.run_dir(base_dir, run_number, tier=model.get_tier())


def _print_nothing_to_continue(
    base_dir: str, model: Model, existing_runs: list[int], run_dir: str
) -> None:
    mismatch = (
        f"was trained against this chain config (tier '{model.get_tier()}') - "
        f"continuing one of them would mix two tiers into a single run"
    )
    print_nothing_to_continue("(R)-(MODIFIER)", len(existing_runs), base_dir, run_dir, mismatch)


def _resolve_run_dir(base_dir: str, model: Model, continue_run: bool) -> tuple[str, bool]:
    """The run dir to work in, and whether it is one being resumed (rather
    than a new one, whose epochs start at 0)."""
    existing_runs = paths.existing_run_numbers(base_dir)

    if continue_run:
        run_number = resume.resumable_run_number(base_dir, model)
        if run_number is not None:
            return paths.run_dir(base_dir, run_number, tier=model.get_tier()), True

        run_dir = _new_run_dir(base_dir, model, existing_runs)
        _print_nothing_to_continue(base_dir, model, existing_runs, run_dir)
        return run_dir, False

    run_dir = _new_run_dir(base_dir, model, existing_runs)
    link_print(
        f"(R)-(MODIFIER) Starting a new run: {run_dir}"
        + (f" (previous run(s) on disk: {existing_runs})" if existing_runs else "")
    )
    return run_dir, False


def _resume_into(model: Model, modifier: ModifierModel, run_dir: str) -> int:
    """Restores `run_dir`'s own prompt and the modifier's history, and says
    which epoch to pick up from."""
    start_epoch = resume.resume_epoch(model, run_dir)
    if start_epoch > 0:
        resume.replay_modifier_history(modifier, run_dir, start_epoch)
        link_print(
            f"(R)-(MODIFIER) Replayed {start_epoch} already-completed epoch(s) "
            f"of {run_dir} into the modifier's history before continuing."
        )
    return start_epoch


def loop(
    epochs: int,
    corpus: Corpus,
    model: Model,
    modifier: ModifierModel,
    continue_run: bool = False,
    val_corpus: Corpus | None = None,
) -> None:
    base_dir = default_base_dir(model.get_type(), model.get_chain_type())

    run_dir, resuming = _resolve_run_dir(base_dir, model, continue_run)
    start_epoch = _resume_into(model, modifier, run_dir) if resuming else 0

    summary_path = reports.write_summary(run_dir)

    total_epochs = epochs

    for e in range(start_epoch, epochs):
        run_epoch(
            e,
            total_epochs,
            corpus,
            model,
            modifier,
            run_dir,
            ask_for_rewrite=True,
            val_corpus=val_corpus,
        )
        summary_path = reports.write_summary(run_dir)
        link_print(f"(R)-(MODIFIER) Summary updated at {summary_path}")
        link_print()

    final_epoch = max(start_epoch, epochs)
    # The final validation pass runs one epoch past the last training epoch,
    # so its own "epoch N/total" display needs total_epochs bumped by one -
    # otherwise it reads as e.g. "4/3".
    final_epoch_total = max(final_epoch + 1, total_epochs)
    if paths.final_epoch_done(run_dir, final_epoch):
        link_print(
            f"(R)-(MODIFIER) Final validation pass (epoch {final_epoch + 1}/"
            f"{final_epoch_total}) already done under {run_dir} - nothing left to "
            f"resume there either. "
            f"Summary at {summary_path} is up to date."
        )
    else:
        run_epoch(
            final_epoch,
            final_epoch_total,
            corpus,
            model,
            modifier,
            run_dir,
            ask_for_rewrite=False,
            val_corpus=val_corpus,
        )
        summary_path = reports.write_summary(run_dir)
        link_print(f"(R)-(MODIFIER) Summary updated at {summary_path}")
        link_print()

    overview_path = reports.write_runs_overview(model)
    if overview_path:
        link_print(
            f"(R)-(MODIFIER) {len(paths.existing_run_numbers(base_dir))} runs now exist "
            f"under {base_dir} - comparison written to {overview_path}"
        )


def _print_startup_banner(
    args: argparse.Namespace, sources: dict[str, str], corpus_path: str, config_path: str | None
) -> None:
    val_file_display = f"'{args.val_file}'" if args.val_file else "none"
    prompt_file_display = f"'{args.prompt_file}'" if args.prompt_file else "none"
    chain_tier_display = f"'{args.chain_tier}'" if args.chain_tier else "chain's own default"

    link_print()
    link_print("(R)-(MODIFIER) Running trainingLoop")
    if config_path:
        link_print(f"(R)-(MODIFIER)   config: '{config_path}'")
    link_print(f"(R)-(MODIFIER)   type: '{args.type}' ({sources['type']})")
    link_print(f"(R)-(MODIFIER)   chain-type: '{args.chain_type}' ({sources['chain_type']})")
    link_print(f"(R)-(MODIFIER)   chain-tier: {chain_tier_display} ({sources['chain_tier']})")
    link_print(f"(R)-(MODIFIER)   file: '{corpus_path}' ({sources['file']})")
    link_print(f"(R)-(MODIFIER)   val-file: {val_file_display} ({sources['val_file']})")
    link_print(f"(R)-(MODIFIER)   prompt-file: {prompt_file_display} ({sources['prompt_file']})")
    link_print(f"(R)-(MODIFIER)   epochs: {args.epochs} ({sources['epochs']})")
    link_print(
        f"(R)-(MODIFIER)   modifier-backend: '{args.modifier_backend}' "
        f"({sources['modifier_backend']})"
    )
    link_print(
        f"(R)-(MODIFIER)   modifier-model: '{args.modifier_model}' ({sources['modifier_model']})"
    )
    link_print(f"(R)-(MODIFIER)   modifier-app: '{args.modifier_app}' ({sources['modifier_app']})")
    link_print(
        f"(R)-(MODIFIER)   modifier-tier: '{args.modifier_tier}' ({sources['modifier_tier']})"
    )
    link_print(f"(R)-(MODIFIER)   continue: {args.continue_run} ({sources['continue_run']})")
    link_print()


def _load_corpora(corpus_path: str, val_file: str | None) -> tuple[Corpus, Corpus | None]:
    link_print()
    link_print("(R)-(MODIFIER) Loading the data...")
    corpus = Corpus()
    corpus.load(corpus_path)

    val_corpus = None
    if val_file:
        val_corpus = Corpus()
        val_corpus.load(val_file)
    link_print("(R)-(MODIFIER) Successfully loaded the data")

    return corpus, val_corpus


def _build_modifier(model: Model, args: argparse.Namespace) -> ModifierModel:
    modifier = ModifierModel(required_placeholders=model.get_prompt_template_vars())
    if args.modifier_backend == "ollama":
        modifier.init_ollama(args.modifier_model)
    else:
        modifier.init_litellm(app_label=args.modifier_app, tier=args.modifier_tier)
    return modifier


if __name__ == "__main__":
    args, sources, config_path = parser.parse_args_with_sources()

    use_generous_llm_timeout()
    setup_django()
    animate_chain()

    corpus_path = args.file or find_corpus_path(args.type)
    if sources["file"] == "default":
        sources["file"] = "auto-search"

    _print_startup_banner(args, sources, corpus_path, config_path)

    model = Model(type=args.type, chain_type=args.chain_type, tier=args.chain_tier)

    try:
        try:
            model.get_system_prompt()
        except AttributeError as e:
            fail(str(e))

        corpus, val_corpus = _load_corpora(corpus_path, args.val_file)
        validate_corpus(model, corpus)
        if val_corpus is not None:
            validate_corpus(model, val_corpus)

        if args.prompt_file:
            apply_prompt_file(model, args.prompt_file, "(R)-(MODIFIER)")

        llm_modifier = _build_modifier(model, args)

        loop(
            args.epochs,
            corpus,
            model,
            llm_modifier,
            continue_run=args.continue_run,
            val_corpus=val_corpus,
        )
    finally:
        model.close()
