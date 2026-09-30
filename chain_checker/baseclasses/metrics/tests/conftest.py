"""Shared fixtures for the metrics test suite.

Entry construction takes four parts - an Input, a true Label, a free-form
info dict and a ModelOutput prediction - and nearly every metric test needs
one, so it is built here once rather than repeated in each test module.
"""

import pytest

from chain_checker.baseclasses.corpus.corpus import Corpus
from chain_checker.baseclasses.corpus.entry import Entry
from chain_checker.baseclasses.corpus.input import Input
from chain_checker.baseclasses.corpus.label import Label
from chain_checker.baseclasses.corpus.output import ModelOutput


@pytest.fixture
def make_entry():
    # A builder, not a single fixture value: a metric only shows its
    # behaviour once several differently-predicted entries have been fed to
    # it, and most tests below build more than one.
    def _make(
        entry_id="case-1",
        *,
        expected=None,
        predicted=None,
        info=None,
        text="hello world",
        token_usage=None,
    ) -> Entry:
        return Entry(
            entry_id,
            Input({"text": text}),
            Label({"passed": True} if expected is None else expected),
            info or {},
            ModelOutput("the conversation", {} if predicted is None else predicted, token_usage),
        )

    return _make


@pytest.fixture
def make_corpus(make_entry):
    def _make(*entries) -> Corpus:
        corpus = Corpus()
        for entry in entries:
            corpus.add_entry(entry)
        return corpus

    return _make
