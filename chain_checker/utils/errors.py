import sys
from typing import NoReturn

from chain_checker.utils.console import print_broken_chain


class CheckerError(SystemExit):
    def __init__(self, message: str) -> None:
        super().__init__(1)
        self.message = message

    def __str__(self) -> str:
        return self.message


def fail(message: str) -> NoReturn:
    print_broken_chain()
    print(f"ERROR: {message}", file=sys.stderr)
    raise CheckerError(message)
