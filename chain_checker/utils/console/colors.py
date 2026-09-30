import os
import re
import sys

IS_TTY = sys.stdout.isatty()
SUPPORTS_COLOR = IS_TTY and os.environ.get("NO_COLOR") is None

RESET = "\x1b[0m" if SUPPORTS_COLOR else ""
DIM = "\x1b[2m" if SUPPORTS_COLOR else ""

LOGO_GRAY = (0xA0, 0xA0, 0xA0)
LOGO_MID_GRAY = (0xD0, 0xD0, 0xD0)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def rgb(color: tuple[int, int, int]) -> str:
    if not SUPPORTS_COLOR:
        return ""
    r, g, b = color
    return f"\x1b[38;2;{r};{g};{b}m"


def visible_width(text: str) -> int:
    return len(_ANSI_RE.sub("", text))
