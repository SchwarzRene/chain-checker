from chain_checker.baseclasses.corpus.corpus import Corpus
from chain_checker.baseclasses.corpus.entry import Entry
from chain_checker.baseclasses.corpus.input import Input
from chain_checker.baseclasses.corpus.label import Label
from chain_checker.baseclasses.corpus.output import ModelOutput
from chain_checker.baseclasses.corpus.output_empty import EmptyModelOutput
from chain_checker.baseclasses.corpus.record import Record

# ModelOutput and EmptyModelOutput are exported from the corpus package
# rather than defined under chain: ModelOutput extends Label, and Corpus
# itself needs EmptyModelOutput to build placeholder predictions in load()
# and reset(). Putting them under chain instead would make corpus and chain
# import from each other.
__all__ = [
    "Corpus",
    "Entry",
    "Input",
    "Label",
    "ModelOutput",
    "EmptyModelOutput",
    "Record",
]
