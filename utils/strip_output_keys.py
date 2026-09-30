import argparse

import yaml

from chain_checker.utils.errors import fail


class _LiteralDumper(yaml.SafeDumper):
    """Its own SafeDumper subclass, so the multi-line string style registered
    below applies only to this dump and not to every yaml.dump() call in the
    process."""


def _str_representer(dumper: yaml.SafeDumper, data: str) -> yaml.Node:
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_LiteralDumper.add_representer(str, _str_representer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--file", required=True, help="Source corpus .yaml to read.")
    parser.add_argument("--out", required=True, help="Path to write the stripped corpus .yaml to.")
    parser.add_argument(
        "--drop",
        required=True,
        nargs="+",
        help="Output key(s) to remove from every case's 'output:' block, e.g. score passed.",
    )
    return parser.parse_args()


def strip_output_keys(data: dict, drop: list[str], path: str) -> dict:
    cases = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(cases, list):
        fail(
            f"Corpus file '{path}' has no top-level 'cases' list.\n"
            f"  A corpus .yaml needs a top-level 'cases: [...]' key."
        )

    for case in cases:
        if not isinstance(case, dict):
            continue
        output = case.get("output")
        if not isinstance(output, dict):
            continue
        for key in drop:
            output.pop(key, None)
        if not output:
            fail(
                f"Corpus file '{path}': case '{case.get('id')}' would end up "
                f"with an empty output.\n"
                f"  Dropped: {drop}\n"
                f"  A case needs at least one checked key left (see "
                f"README.md's corpus section)."
            )

    return data


def main() -> None:
    args = parse_args()

    try:
        with open(args.file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except OSError as e:
        fail(f"Could not open corpus file '{args.file}'.\n  Reason: {e.strerror or e}")

    except yaml.YAMLError as e:
        fail(f"Corpus file '{args.file}' is not valid YAML.\n  Reason: {e}")

    data = strip_output_keys(data, args.drop, args.file)

    with open(args.out, "w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            Dumper=_LiteralDumper,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=1000,
        )

    print(f"Wrote '{args.out}' - dropped {args.drop} from every case's output.")


if __name__ == "__main__":
    main()
