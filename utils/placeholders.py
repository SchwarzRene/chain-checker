import re
from collections.abc import Sequence
from string import Formatter

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


_FIELD_SUFFIX_RE = re.compile(r"[.\[:!]")


def field_root(field_name: str) -> str | None:
    root = _FIELD_SUFFIX_RE.split(field_name, maxsplit=1)[0]
    return root if _IDENTIFIER_RE.match(root) else None


def detect_placeholders(text: str) -> tuple[str, ...]:
    names = []
    for _, field_name, _, _ in Formatter().parse(text):
        if field_name is None:
            continue
        root = field_root(field_name)
        if root is not None:
            names.append(root)

    return tuple(dict.fromkeys(names))


def escape_stray_braces(text: str, template_vars: Sequence[str]) -> str:
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        char = text[i]

        if char == "{":
            field = _declared_field_at(text, i, template_vars)
            if field is not None:
                out.append(field)
                i += len(field)
                continue
            out.append("{{")
        elif char == "}":
            out.append("}}")
        else:
            out.append(char)

        i += 1

    return "".join(out)


def _declared_field_at(text: str, start: int, template_vars: Sequence[str]) -> str | None:
    end = text.find("}", start)
    if end == -1:
        return None

    root = field_root(text[start + 1 : end])
    if root is not None and root in template_vars:
        return text[start : end + 1]
    return None


_NORMALIZE_RE = re.compile(r"[\s_-]")


def _normalize(name: str) -> str:
    return _NORMALIZE_RE.sub("", name).lower()


def find_near_miss_placeholders(text: str, declared_vars: Sequence[str]) -> tuple[str, ...]:
    # A declared var with a `{...}` span that nearly names it (stray spacing,
    # a hyphen instead of `_`) almost certainly meant to reference it and
    # broke - a real mistake, not a choice. A declared var with no brace at
    # all referencing it anywhere is a clean freeze/bake-in instead (see
    # CHAIN-REQUIREMENTS.md's placeholder section) - not flagged here.
    normalized_declared = {_normalize(v): v for v in declared_vars}

    hits: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue

        end = text.find("}", i + 1)
        if end == -1:
            break

        inner = text[i + 1 : end].strip()
        root = field_root(inner)
        candidate = root if root is not None else inner
        var = normalized_declared.get(_normalize(candidate))
        if var is not None and var not in hits:
            hits.append(var)

        i = end + 1

    return tuple(hits)


def unescape_braces(text: str) -> str:
    out = []
    i, n = 0, len(text)
    while i < n:
        out.append(text[i])
        i += 2 if text.startswith("{{", i) or text.startswith("}}", i) else 1

    return "".join(out)
