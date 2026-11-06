"""Chat-client backends for the free Qwen frontier labeler/judge.

Two interchangeable backends, both exposing `chat()` and `chat_json()`:

  * :class:`OpenAICompatClient` -- any OpenAI-compatible /v1/chat/completions
    endpoint (standard library only, no `openai` SDK):
      - Ollama (local, free, no key):  base_url=http://localhost:11434/v1
      - OpenRouter (free tier, key):   base_url=https://openrouter.ai/api/v1
      - vLLM / LM Studio / etc.
  * :class:`TransformersChatClient` -- a Qwen instruct model loaded locally via
    transformers with 4-bit quantization, for a **Colab GPU** run. The heavy
    model is **lazy-loaded on first use**, so constructing the client (and the
    labeler) needs no torch -- offline tests stay green.

JSON parsing is lenient: we extract the first {...} object from the reply,
since small local models sometimes wrap JSON in prose.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


class LLMError(RuntimeError):
    pass


def parse_json_reply(raw: str) -> Dict[str, Any]:
    """Lenient parse: whole string as JSON, else the first {...} object in it."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        m = _JSON_OBJ.search(raw or "")
        if not m:
            raise LLMError(f"no JSON object found in reply: {raw!r}")
        return json.loads(m.group(0))


@dataclass
class OpenAICompatClient:
    base_url: str = "http://localhost:11434/v1"   # Ollama default
    model: str = "qwen2.5:3b"
    api_key_env: str = "OPENAI_API_KEY"           # Ollama ignores the key
    temperature: float = 0.0
    max_tokens: int = 256
    timeout_s: int = 60
    force_json: bool = True                         # request JSON response_format

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env, "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def chat(self, system: str, user: str) -> str:
        """Return the assistant message content as a string."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.force_json:
            payload["response_format"] = {"type": "json_object"}

        data = json.dumps(payload).encode("utf-8")
        url = self.base_url.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(url, data=data, headers=self._headers(),
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:  # connection refused, timeout, etc.
            raise LLMError(f"LLM endpoint {url} unreachable: {e}") from e
        except json.JSONDecodeError as e:
            raise LLMError(f"LLM returned non-JSON envelope from {url}") from e
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected LLM response shape: {body}") from e

    def chat_json(self, system: str, user: str) -> Dict[str, Any]:
        """chat() + lenient parse of the first {...} object in the reply."""
        return parse_json_reply(self.chat(system, user))

    def ping(self) -> bool:
        """True if the endpoint answers a trivial request."""
        try:
            self.chat("Reply with JSON.", 'Return {"ok": true}.')
            return True
        except LLMError:
            return False


class TransformersChatClient:
    """Qwen instruct model via transformers (4-bit), for a Colab GPU run.

    The model is loaded **lazily on first `chat()`** by reusing
    `swarm.worker.HFModel`, so importing/constructing this client requires no
    torch -- offline tests that never call it stay green.
    """

    def __init__(self, model: str = "Qwen/Qwen2.5-3B-Instruct",
                 quantization: str = "4bit", max_new_tokens: int = 256,
                 temperature: float = 0.0, seed: int = 0):
        self.model = model
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.seed = seed
        self._hf = None  # lazy

    def _ensure(self):
        if self._hf is None:
            from swarm.worker import HFModel  # heavy import, deferred
            self._hf = HFModel(
                model_name=self.model, quantization=self.quantization,
                max_new_tokens=self.max_new_tokens, temperature=self.temperature,
                seed=self.seed,
            )
        return self._hf

    def chat(self, system: str, user: str) -> str:
        try:
            return self._ensure().generate(system, user)
        except Exception as e:  # OOM, load failure, decode failure
            raise LLMError(f"transformers chat failed: {e}") from e

    def chat_json(self, system: str, user: str) -> Dict[str, Any]:
        return parse_json_reply(self.chat(system, user))

    def ping(self) -> bool:
        try:
            self.chat("Reply with JSON.", 'Return {"ok": true}.')
            return True
        except LLMError:
            return False


def build_chat_client(block: Dict[str, Any]):
    """Build a chat client from a config sub-block (e.g. cfg['labeling']['qwen']).

    `backend`: "transformers" (Colab GPU, default) | "openai" (Ollama/OpenRouter).
    `model` is an HF repo id for the transformers backend, or an Ollama/OpenRouter
    model tag for the openai backend.
    """
    backend = block.get("backend", "transformers")
    if backend == "transformers":
        return TransformersChatClient(
            model=block.get("model", "Qwen/Qwen2.5-3B-Instruct"),
            quantization=block.get("quantization", "4bit"),
            max_new_tokens=block.get("max_new_tokens", 256),
            temperature=block.get("temperature", 0.0),
        )
    if backend == "openai":
        return OpenAICompatClient(
            base_url=block.get("base_url", "http://localhost:11434/v1"),
            model=block.get("model", "qwen2.5:3b"),
            api_key_env=block.get("api_key_env", "OPENAI_API_KEY"),
            temperature=block.get("temperature", 0.0),
            max_tokens=block.get("max_new_tokens", 256),
            timeout_s=block.get("timeout_s", 60),
            force_json=block.get("force_json", True),
        )
    raise ValueError(f"unknown qwen backend {backend!r} (want 'transformers' or 'openai')")


# backward-compat alias
def client_from_cfg(block: Dict[str, Any]):
    return build_chat_client(block)
