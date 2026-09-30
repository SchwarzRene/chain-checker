"""Escapes quotes a model left unescaped inside a JSON string value.

    {"evidence": "All correct: "100% ACME", "ACME Partner" - fine.",
     "passed": true}

is the answer that was asked for and still unparseable: the decoder ends
`evidence` at the quote before `100%`. Patching the text keeps the answer
instead of dropping the case and re-prompting for it.

Not a JSON parser: it only decides where each string ends, never what the
document means, and leaves accepting or rejecting it to the real decoder.
"""

_WHITESPACE = " \t\r\n"


def _skip_whitespace(text: str, i: int) -> int:
    while i < len(text) and text[i] in _WHITESPACE:
        i += 1
    return i


def _end_of_quoted(text: str, i: int) -> int:
    """Index just past the closing quote of the string opening at `i`, or -1
    if it never closes."""
    i += 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == '"':
            return i + 1
        i += 1
    return -1


def _next_member_follows(text: str, i: int) -> bool:
    """Whether `text` from `i` reads as `"..." :`, the start of the next
    object member."""
    i = _skip_whitespace(text, i)
    if i >= len(text) or text[i] != '"':
        return False
    end = _end_of_quoted(text, i)
    if end < 0:
        return False
    end = _skip_whitespace(text, end)
    return end < len(text) and text[end] == ":"


def _closes_the_string(text: str, i: int, *, is_key: bool, in_object: bool) -> bool:
    """Whether the unescaped quote at `i` really ends its string.

    A closing quote is only ever followed by structure, so anything else
    means the quote belongs to the text. The hard case is a stray quote
    followed by a comma, which looks exactly like a terminator:

        "evidence": "correct: "100% ACME", "ACME Partner" - fine.",
                                          ^ stray            ^ stray  ^ real

    Only the real one is followed by a comma and the next `"key":` pair.
    """
    after = _skip_whitespace(text, i + 1)
    if after >= len(text):
        return True

    char = text[after]
    if is_key:
        return char == ":"
    if char in "}]":
        return True
    if char != ",":
        return False
    # In an array the comma just introduces the next element, of any shape.
    return _next_member_follows(text, after + 1) if in_object else True


def _copy_string(text: str, i: int, out: list[str], *, is_key: bool, in_object: bool) -> int:
    """Copies the string opening at `i` into `out`, escaping every quote in it
    that is not the terminator. Returns the index just past the string."""
    out.append('"')
    i += 1
    while i < len(text):
        char = text[i]
        if char == "\\":
            # Copy the pair through so a legal \" is not double-escaped.
            out.append(text[i : i + 2])
            i += 2
            continue
        if char == '"':
            if _closes_the_string(text, i, is_key=is_key, in_object=in_object):
                out.append('"')
                return i + 1
            out.append('\\"')
            i += 1
            continue
        out.append(char)
        i += 1
    return i


def escape_stray_quotes(text: str) -> str:
    """Returns `text` with unescaped quotes inside its JSON string literals
    escaped, and everything else byte-identical. Only ever inserts
    backslashes, so no content can be lost. Text it cannot repair — prose, a
    string that never closes — comes back unchanged rather than raising."""
    out: list[str] = []
    containers: list[str] = []
    expecting_key = False
    i = 0

    while i < len(text):
        char = text[i]
        if char == '"':
            in_object = bool(containers) and containers[-1] == "{"
            i = _copy_string(text, i, out, is_key=in_object and expecting_key, in_object=in_object)
            continue

        if char in "{[":
            containers.append(char)
            expecting_key = char == "{"
        elif char in "}]":
            if containers:
                containers.pop()
            expecting_key = bool(containers) and containers[-1] == "{"
        elif char in ",:":
            # A comma in an object starts the next member, so a key follows.
            expecting_key = char == "," and bool(containers) and containers[-1] == "{"

        out.append(char)
        i += 1

    return "".join(out)
