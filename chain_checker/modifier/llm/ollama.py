import json
import sys
import urllib.error
import urllib.request
from typing import Any

from chain_checker.modifier.llm.llm_baseclass import LLM
from chain_checker.utils.console import link_print
from chain_checker.utils.errors import fail

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT = 10800
DEFAULT_NUM_CTX = 256000
DEFAULT_NUM_PREDICT = -1


class Ollama(LLM):
    """Modifier backend that talks to a local `ollama serve` instance over
    its streaming NDJSON `/api/generate` endpoint."""

    def __init__(
        self,
        model: str,
        host: str = DEFAULT_HOST,
        timeout: float = DEFAULT_TIMEOUT,
        num_ctx: int = DEFAULT_NUM_CTX,
        num_predict: int = DEFAULT_NUM_PREDICT,
    ) -> None:
        super().__init__()
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._num_ctx = num_ctx
        self._num_predict = num_predict

    def get_run_info(self) -> dict[str, str]:
        return {"backend": "ollama", "model": self._model}

    def _build_request(self, prompt: str) -> urllib.request.Request:
        body = json.dumps(
            {
                "model": self._model,
                "prompt": prompt,
                "stream": True,
                # Streams the model's reasoning as "thinking" chunks alongside its
                # answer, printed live in _stream_response below.
                "think": True,
                "options": {
                    "num_ctx": self._num_ctx,
                    "num_predict": self._num_predict,
                },
            }
        ).encode("utf-8")

        return urllib.request.Request(
            f"{self._host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )

    def _stream_response(self, req: urllib.request.Request) -> tuple[list[str], dict[str, Any]]:
        response_parts: list[str] = []
        final_chunk: dict[str, Any] = {}
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                for line in resp:
                    line = line.strip()
                    if not line:
                        continue

                    chunk = json.loads(line)
                    if "error" in chunk:
                        fail(f"Ollama returned an error.\n  Reason: {chunk['error']}")

                    text = chunk.get("thinking") or chunk.get("response") or ""
                    if text:
                        # link_print prefixes every *line*, which is fine once per
                        # message but chops a token stream mid-sentence - write
                        # tokens straight to stdout instead.
                        sys.stdout.write(text)
                        sys.stdout.flush()
                    if chunk.get("response"):
                        response_parts.append(chunk["response"])

                    if chunk.get("done"):
                        final_chunk = chunk
        except urllib.error.URLError as e:
            fail(
                f"Could not reach Ollama at {self._host}.\n"
                f"  Reason: {e}\n"
                f"  Is `ollama serve` running?"
            )
        finally:
            # The loop above prints without a trailing newline.
            link_print()

        return response_parts, final_chunk

    def __call__(self, prompt: str) -> str:
        req = self._build_request(prompt)
        response_parts, final_chunk = self._stream_response(req)

        self._record_usage(
            final_chunk.get("prompt_eval_count", 0) or 0,
            final_chunk.get("eval_count", 0) or 0,
        )

        response = "".join(response_parts)
        return self._require_nonempty_text(
            response,
            "Ollama generated no 'response' content at all.\n"
            f"  done_reason={final_chunk.get('done_reason', '?')}\n"
            "  The model may have run out of tokens before producing an "
            "answer - try a larger num_predict.",
        )


if __name__ == "__main__":
    llm = Ollama(model="qwen3.5:2b")
    link_print(llm("Say hello in one short sentence."))
