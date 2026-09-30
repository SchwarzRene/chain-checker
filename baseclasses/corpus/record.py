from typing import Any

_NO_DEFAULT = object()


class Record:
    """Base for the corpus's flat named-field data holders (`Input`,
    `Label`, `ModelOutput`). Just wraps a dict with a small read API - the
    subclasses exist to distinguish what role the same shape plays."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._replace(data)

    def _replace(self, data: dict[str, Any]) -> None:
        self._data: dict[str, Any] = dict(data)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}:\n{self._format_data(self._data)}"

    def get(self) -> dict[str, Any]:
        return dict(self._data)

    def get_keys(self) -> list[str]:
        return list(self._data.keys())

    def get_value(self, key: str, default: Any = _NO_DEFAULT) -> Any:
        # A sentinel, not None, marks "no default given": None is a
        # legitimate default a caller might pass, so it can't double as
        # "raise KeyError instead."
        if default is _NO_DEFAULT:
            return self._data[key]
        return self._data.get(key, default)

    @staticmethod
    def _format_data(data: dict[str, Any]) -> str:
        return "\n".join(f"  {key}: {value}" for key, value in data.items())
