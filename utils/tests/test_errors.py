"""Unit tests for chain_checker/utils/errors.py.

fail() is in the corpus merge because Corpus.load() and its validation
helpers route every rejection through it. Its contract is narrow and every
caller depends on all of it: print the reason, then end the process - so
the code after a fail() call never runs.

It ends the process by raising CheckerError, a SystemExit subclass, rather
than calling sys.exit directly. The tests below cover both halves of why:
it still has to unwind past the broad `except Exception` handlers the run is
built out of, and it now has to be catchable on purpose by a caller that
hosts the corpus layer in a longer-lived process.
"""

import pytest

from chain_checker.utils.errors import CheckerError, fail


def test_fail_ends_the_program():
    with pytest.raises(SystemExit):
        fail("something broke")


def test_fail_exits_non_zero():
    # A shell or CI job running checker.py has to be able to tell a rejected
    # corpus from a passing run.
    with pytest.raises(SystemExit) as exit_info:
        fail("something broke")

    assert exit_info.value.code == 1


def test_fail_prints_the_reason_with_an_error_prefix(capsys):
    with pytest.raises(SystemExit):
        fail("corpus file 'x.yaml' has no top-level 'cases' list")

    assert capsys.readouterr().err == (
        "ERROR: corpus file 'x.yaml' has no top-level 'cases' list\n"
    )


def test_fail_prints_to_stderr_not_stdout(capsys):
    # This used to be pinned the other way round, on the grounds that the
    # message is the user-facing diagnostic. It is - which is the argument for
    # stderr, not against it. checker.py and trainingLoop.py stream progress
    # to stdout, so on stdout a fatal error was interleaved with normal
    # output, and `python bin/checker.py > run.log` swallowed it outright:
    # the log held the run, stderr held nothing, and CI reported a bare
    # exit code 1 with no reason attached to it.
    with pytest.raises(SystemExit):
        fail("a reason")

    captured = capsys.readouterr()

    assert "a reason" in captured.err
    # The broken-chain notice (below) is decorative and goes to stdout, but
    # the actual reason must never be duplicated there.
    assert "a reason" not in captured.out


def test_fail_prints_the_broken_chain_notice_before_the_reason(capsys):
    # Every fail() call means the run cannot continue, so every one gets the
    # same visual - a caller never has to remember to print it themselves.
    with pytest.raises(SystemExit):
        fail("a reason")

    assert "Looks like the chain is broken" in capsys.readouterr().out


def test_fail_does_not_return_to_its_caller():
    # The invariant Corpus._check_case relies on: it calls fail() and then
    # returns False, and load() honours that False - but nothing downstream
    # of a fail() call is reachable. Annotated NoReturn so mypy enforces it
    # rather than leaving it to be rediscovered.
    reached_after_fail = False

    def caller() -> None:
        nonlocal reached_after_fail
        fail("stop here")
        reached_after_fail = True  # unreachable

    with pytest.raises(SystemExit):
        caller()

    assert reached_after_fail is False


def test_fail_accepts_a_multiline_message(capsys):
    # Corpus's diagnostics are long, wrapped f-strings.
    with pytest.raises(SystemExit):
        fail("first line\nsecond line")

    assert capsys.readouterr().err == "ERROR: first line\nsecond line\n"


def test_fail_accepts_an_empty_message(capsys):
    with pytest.raises(SystemExit):
        fail("")

    assert capsys.readouterr().err == "ERROR: \n"


# --------------------------------------------------------------------------
# CheckerError: the exit has a name, and keeps sys.exit's reach
# --------------------------------------------------------------------------


def test_a_broad_except_exception_does_not_swallow_a_failure():
    # The reason CheckerError derives from SystemExit rather than Exception.
    # loop/center.py wraps the model call in `except Exception` and counts a
    # raising call as one failed entry so a training run can finish the
    # corpus; chain/model.py wraps generateSystemPrompt in `except Exception`
    # and falls back to the raw prompt. If a fatal error were an Exception,
    # both would downgrade it silently - a chain that isn't registered would
    # be reported as a single bad case, and the run would carry on scoring
    # against a corpus it could never satisfy.
    swallowed = False

    def caller() -> None:
        nonlocal swallowed
        try:
            fail("chain 'x' is not registered")
        except Exception:  # noqa: BLE001 - the point of the test
            swallowed = True

    with pytest.raises(SystemExit):
        caller()

    assert swallowed is False


def test_a_caller_that_wants_to_can_catch_the_failure_by_name():
    # What the named type buys: Corpus.load() is library code, reachable in
    # this backend from a management command, a request handler or a task
    # worker. Those can now recover from a bad .yaml instead of having the
    # process taken down under them.
    try:
        fail("corpus file 'x.yaml' is not valid YAML")
    except CheckerError as e:
        assert str(e) == "corpus file 'x.yaml' is not valid YAML"
        assert e.message == "corpus file 'x.yaml' is not valid YAML"
    else:
        pytest.fail("fail() should have raised CheckerError")


def test_the_failure_is_still_a_systemexit_for_anything_that_only_knows_that():
    # Nothing is obliged to learn the new name - every caller and test that
    # already expected SystemExit keeps working.
    with pytest.raises(CheckerError) as exit_info:
        fail("something broke")

    assert isinstance(exit_info.value, SystemExit)
    assert not isinstance(exit_info.value, Exception)


def test_str_renders_the_message_rather_than_the_exit_code():
    # SystemExit.__str__ renders its code, so an un-overridden CheckerError
    # would log and re-raise as the string "1".
    assert str(CheckerError("the reason")) == "the reason"
