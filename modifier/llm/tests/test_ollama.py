import json
import urllib.error
from collections.abc import Iterator
from unittest.mock import patch

import pytest

from chain_checker.modifier.llm.ollama import Ollama
from chain_checker.utils.errors import CheckerError


class _FakeStream:
    """Stands in for the response `urllib.request.urlopen` hands back: a
    context manager whose iteration yields one raw NDJSON line per Ollama
    chunk, exactly like the real streaming body."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __enter__(self) -> "_FakeStream":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def __iter__(self) -> "Iterator[bytes]":
        return iter(self._lines)


def _served(chunks: list[dict]):
    lines = [json.dumps(chunk).encode() for chunk in chunks]
    return patch("urllib.request.urlopen", return_value=_FakeStream(lines))


def _served_raw(lines: list[bytes]):
    return patch("urllib.request.urlopen", return_value=_FakeStream(lines))


def test_get_run_info_names_the_backend_and_model():
    assert Ollama(model="qwen3.5:4b").get_run_info() == {"backend": "ollama", "model": "qwen3.5:4b"}


def test_build_request_targets_the_generate_endpoint_with_the_configured_options():
    llm = Ollama(model="qwen3.5:4b", host="http://example:11434/", num_ctx=123, num_predict=7)

    req = llm._build_request("hello")

    assert req.full_url == "http://example:11434/api/generate"
    body = json.loads(req.data)
    assert body["model"] == "qwen3.5:4b"
    assert body["prompt"] == "hello"
    assert body["options"] == {"num_ctx": 123, "num_predict": 7}


def test_a_trailing_slash_on_the_host_is_not_doubled():
    llm = Ollama(model="m", host="http://example:11434/")

    assert llm._build_request("hi").full_url == "http://example:11434/api/generate"


def test_the_streamed_response_text_is_assembled_in_order():
    with _served(
        [
            {"response": "Hel"},
            {"response": "lo"},
            {"done": True, "prompt_eval_count": 3, "eval_count": 2},
        ]
    ):
        llm = Ollama(model="m")
        assert llm("hi") == "Hello"
        assert llm.get_last_usage() == {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }


def test_a_blank_line_in_the_stream_is_skipped():
    with _served_raw(
        [
            b"   ",
            json.dumps({"response": "ok"}).encode(),
            json.dumps({"done": True}).encode(),
        ]
    ):
        assert Ollama(model="m")("hi") == "ok"


def test_an_error_chunk_fails_the_call():
    with _served([{"error": "model not found"}]):
        with pytest.raises(CheckerError, match="model not found"):
            Ollama(model="m")("hi")


def test_an_empty_reply_fails_rather_than_return_nothing():
    with _served([{"done": True, "done_reason": "length"}]):
        with pytest.raises(CheckerError, match="done_reason=length"):
            Ollama(model="m")("hi")


def test_an_unreachable_host_fails_with_a_clear_reason():
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        with pytest.raises(CheckerError, match="Is `ollama serve` running"):
            Ollama(model="m")("hi")


def test_thinking_text_counts_toward_neither_response_nor_usage():
    # "thinking" chunks are printed live but never appended to the reply.
    with _served(
        [
            {"thinking": "pondering..."},
            {"response": "done"},
            {"done": True, "prompt_eval_count": 1, "eval_count": 1},
        ]
    ):
        assert Ollama(model="m")("hi") == "done"
