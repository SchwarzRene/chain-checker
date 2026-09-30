import glob
import os

from chain_checker.baseclasses.chain.model import Model as Chain
from chain_checker.baseclasses.corpus import Corpus
from chain_checker.utils.errors import fail


def find_corpus_path(type: str) -> str:
    matches = glob.glob(os.path.join("workflows", type, "*.yaml"))
    if not matches:
        fail(
            f"No .yaml corpus file found in workflows/{type}.\n"
            f"  Add one under workflows/{type}/, or pass --file to point "
            f"at one explicitly."
        )
    if len(matches) > 1:
        fail(
            f"Multiple .yaml corpus files found in workflows/{type}.\n"
            f"  Found: {matches}\n"
            f"  Pass --file to pick one explicitly."
        )
    return matches[0]


def validate_corpus(model: Chain, corpus: Corpus) -> None:
    chain = model.chain

    if not hasattr(chain, "OutputSchema"):
        fail(
            f"Chain '{model.get_chain_type()}' has no OutputSchema.\n"
            f"  Every chain needs one so a corpus case's output keys can "
            f"be checked against it (see README.md's chain section)."
        )

    input_fields = chain.InputSchema.model_fields
    required_input_fields = {name for name, field in input_fields.items() if field.is_required()}
    output_fields = set(chain.OutputSchema.model_fields.keys())

    for entry in corpus:
        input_data = entry.get_input().get()
        unknown = [key for key in input_data if key not in input_fields]
        missing = [key for key in required_input_fields if key not in input_data]
        if unknown or missing:
            detail_lines = [
                f"  Found key(s): {sorted(input_data, key=str)}",
                f"  Expected field(s): {sorted(input_fields)}",
            ]
            if unknown:
                detail_lines.append(f"  Unknown key(s): {unknown}")
            if missing:
                detail_lines.append(f"  Missing required key(s): {missing}")
            detail_lines.append("  See README.md's chain section.")
            fail(
                f"Corpus case '{entry.get_id()}': input does not match "
                f"chain '{model.get_chain_type()}'s InputSchema.\n" + "\n".join(detail_lines)
            )

        try:
            chain.InputSchema(**input_data)
        except Exception as e:
            fail(
                f"Corpus case '{entry.get_id()}': input does not match "
                f"chain '{model.get_chain_type()}'s InputSchema.\n"
                f"  Found key(s): {sorted(input_data, key=str)}\n"
                f"  Reason: {e}\n"
                f"  See README.md's chain section."
            )

        unknown_output = [key for key in entry.get_output().get_keys() if key not in output_fields]
        if unknown_output:
            fail(
                f"Corpus case '{entry.get_id()}': output key(s) are not "
                f"part of chain '{model.get_chain_type()}'s OutputSchema.\n"
                f"  Unknown: {unknown_output}\n"
                f"  Chain output field(s): {sorted(output_fields)}\n"
                f"  A case's output can only check keys the chain's real "
                f"output actually has (see README.md's chain section)."
            )
