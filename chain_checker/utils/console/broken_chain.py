from chain_checker.utils.console.colors import LOGO_MID_GRAY, RESET, rgb
from chain_checker.utils.console.link_print import link_print

# A chain-link glyph (matching link_print's own "  O   O  " pattern)
# collapsing in on itself and unravelling into loose links - printed right
# before an error that means a chain could not be loaded at all, so the
# failure reads as part of the tool's own visual language instead of a bare
# traceback.
_BROKEN_CHAIN_ART = "  O   O\n   cↄ 0\n    cↄ 0\n      cↄ0\n         cↄ-cↄ-cↄ-cↄ-c\n \n"


def print_broken_chain() -> None:
    link_print("(CHECKER) Looks like the chain is broken...")
    color = rgb(LOGO_MID_GRAY)
    for line in _BROKEN_CHAIN_ART.split("\n"):
        print(f"{color}{line}{RESET}")
