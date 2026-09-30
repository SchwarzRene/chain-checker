# ruff: noqa: E402
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from chain_checker.baseclasses.chain.calls import render_conversation, sum_token_usage
from chain_checker.baseclasses.chain.model import ChainRunError, Model as Chain
from chain_checker.baseclasses.corpus import Corpus
from chain_checker.baseclasses.corpus.output import ModelOutput
from chain_checker.baseclasses.loop.cache import LoopCache
from chain_checker.baseclasses.loop.progress import ProgressTracker
from chain_checker.utils.bootstrap import setup_django, use_generous_llm_timeout
from chain_checker.utils.checker_utils.cli import parse_args
from chain_checker.utils.checker_utils.report import (
    existing_check_dirs,
    next_check_dir,
    resumable_check_dir,
    write_report,
)
from chain_checker.utils.checker_utils.startup import (
    apply_prompt_file,
    print_nothing_to_continue,
)
from chain_checker.utils.checker_utils.validation import find_corpus_path, validate_corpus
from chain_checker.utils.console import (
    animate_chain,
    link_print,
    link_print_warning,
    next_glyph,
)


def _case_prefix(model: Chain, entry, index: int, total: int) -> str:
    return (
        f"{next_glyph()} (CHECKER) case ({index})/({total}) '{entry.get_id()}' -> "
        f"waiting on chain '{model.get_chain_type()}'"
    )


def run_case(
    model: Chain, entry, index: int, total: int, tracker: ProgressTracker
) -> tuple[bool, Exception | None, float]:
    prefix = _case_prefix(model, entry, index, total)
    predicted, error, elapsed = model.call_with_progress(entry.get_input(), prefix, tracker)

    if error is not None:
        _print_case_failure(entry, error)
        _record_failed_conversation(entry, error)
        return False, error, elapsed

    entry.set_model_output(predicted)

    expected = entry.get_output()
    matches = expected.matches(predicted)
    return all(matches.values()), None, elapsed


def _print_case_failure(entry, error: Exception) -> None:
    link_print_warning(
        f"(CHECKER) WARNING: case '{entry.get_id()}' failed: {error} - counting "
        f"it as an empty prediction and continuing with the rest of the corpus."
    )


def _record_failed_conversation(entry, error: Exception) -> None:
    if not isinstance(error, ChainRunError) or not error.calls:
        return

    conversation = render_conversation(error.calls)
    conversation += _after_stripping_section(error)
    entry.set_model_output(ModelOutput(conversation, {}, sum_token_usage(error.calls)))


def _after_stripping_section(error: ChainRunError) -> str:
    cleaned_reply = getattr(error.__cause__, "cleaned_reply", None)
    if cleaned_reply is None:
        return ""
    return f"--- after stripping ---\n{cleaned_reply}\n\n"


def _print_output_shape_suggestion(failed: int, total: int, last_error: Exception | None) -> None:
    if total == 0 or failed < total:
        return

    link_print(
        f"(CHECKER) every case failed to run; last error: {type(last_error).__name__}: {last_error}"
    )


def _resolve_check_dir(model: Chain, continue_run: bool) -> str:
    if not continue_run:
        return next_check_dir(model)

    resumable = resumable_check_dir(model)
    if resumable is not None:
        link_print(f"(CHECKER) --continue: recalling cached predictions from {resumable}")
        return resumable

    fresh = next_check_dir(model)
    _print_nothing_to_continue(model, fresh)
    return fresh


def _print_nothing_to_continue(model: Chain, fresh: str) -> None:
    existing = existing_check_dirs(model)
    mismatch = (
        f"was made with this run's prompt and chain config (tier '{model.get_tier()}') "
        f"- the newest, '{os.path.basename(existing[0])}', differs"
        if existing
        else ""
    )
    print_nothing_to_continue("(CHECKER)", len(existing), os.path.dirname(fresh), fresh, mismatch)


def _save_failed_case(cache: LoopCache, entry) -> None:
    """Keeps a failed case's raw prompt/reply for reading, but never as a
    cached prediction - same rule as ClassificationLoop._run_entry, which
    saves nothing at all when an entry raises.
    """
    if not entry.get_model_output().get_convo():
        return

    cache.save_failure(entry.get_id(), entry.get_model_output())
    link_print(
        f"(CHECKER)   raw prompt/reply for this failed case saved to "
        f"{cache.get_failure_path(entry.get_id())} - it stays out of the "
        f"cache, so --continue retries this case instead of recalling it."
    )


def _score_recalled(entry) -> bool:
    expected = entry.get_output()
    matches = expected.matches(entry.get_model_output())
    return all(matches.values())


def run(
    type: str,
    chain_type: str,
    corpus_path: str,
    prompt_file: str | None = None,
    chain_tier: str | None = None,
    continue_run: bool = False,
) -> bool:
    corpus = Corpus()
    corpus.load(corpus_path)

    model = Chain(type=type, chain_type=chain_type, tier=chain_tier)

    try:
        if prompt_file:
            apply_prompt_file(model, prompt_file, "(CHECKER)")

        validate_corpus(model, corpus)

        check_dir = _resolve_check_dir(model, continue_run)
        base_dir, run_id = os.path.split(check_dir)
        cache = LoopCache(run_id, corpus, model, base_dir=base_dir)

        # Recall must run before save_prompt()/save_config() overwrite the very
        # files it compares the current prompt/config against - same order as
        # ClassificationLoop.loop().
        recalled = cache.recall()
        cache.save_prompt()
        cache.save_config()

        total = len(corpus)
        if recalled:
            link_print(
                f"(CHECKER) {len(recalled)}/{total} case(s) recalled from {cache.get_entries_dir()}"
            )

        outcomes: list[tuple[bool, Exception | None]] = []
        failed_ids: set[str | int] = set()
        tracker = ProgressTracker(total - len(recalled))
        for index, entry in enumerate(corpus, start=1):
            if str(entry.get_id()) in recalled:
                outcomes.append((_score_recalled(entry), None))
                continue

            ok, error, _ = run_case(model, entry, index, total, tracker)

            if error is None:
                cache.save_prediction(entry.get_id(), entry.get_model_output())
            else:
                failed_ids.add(entry.get_id())
                _save_failed_case(cache, entry)
            outcomes.append((ok, error))

        passed = sum(1 for ok, _ in outcomes if ok)
        failed = sum(1 for _, error in outcomes if error is not None)
        last_error = next((error for _, error in reversed(outcomes) if error is not None), None)

        link_print()
        link_print(f"(CHECKER) {passed}/{total} cases passed")
        _print_output_shape_suggestion(failed, total, last_error)

        report_path = write_report(model, corpus, check_dir, failed_ids=failed_ids)
        link_print(f"(CHECKER)-(REPORT) Report written to {report_path}")

        return passed == total
    finally:
        model.close()


def _print_startup_banner(
    args: argparse.Namespace, sources: dict[str, str], corpus_path: str, config_path: str | None
) -> None:
    prompt_file_display = f"'{args.prompt_file}'" if args.prompt_file else "none"
    chain_tier_display = f"'{args.chain_tier}'" if args.chain_tier else "chain's own default"

    link_print("")
    link_print("(CHECKER) Running checker")
    if config_path:
        link_print(f"(CHECKER)   config: '{config_path}'")
    link_print(f"(CHECKER)   type: '{args.type}' ({sources['type']})")
    link_print(f"(CHECKER)   chain-type: '{args.chain_type}' ({sources['chain_type']})")
    link_print(f"(CHECKER)   chain-tier: {chain_tier_display} ({sources['chain_tier']})")
    link_print(f"(CHECKER)   file: '{corpus_path}' ({sources['file']})")
    link_print(f"(CHECKER)   prompt-file: {prompt_file_display} ({sources['prompt_file']})")
    link_print(f"(CHECKER)   continue: {args.continue_run} ({sources['continue_run']})")
    link_print()


if __name__ == "__main__":
    args, sources, config_path = parse_args()

    use_generous_llm_timeout()
    setup_django()
    animate_chain()

    corpus_path = args.file or find_corpus_path(args.type)
    if sources["file"] == "default":
        sources["file"] = "auto-search"

    _print_startup_banner(args, sources, corpus_path, config_path)

    ok = run(
        args.type,
        args.chain_type,
        corpus_path,
        args.prompt_file,
        args.chain_tier,
        continue_run=args.continue_run,
    )

    sys.exit(0 if ok else 1)
