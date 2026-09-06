"""Pluggable LLM chat backends.

Three implementations share one interface:

* :class:`OpenAIBackend`     -- OpenAI Chat Completions (key from ``OPENAI_API_KEY``)
* :class:`AnthropicBackend`  -- Anthropic Messages API (key from ``ANTHROPIC_API_KEY``)
* :class:`MockBackend`       -- deterministic, offline, scriptable

API keys are read from environment variables only; they are never accepted as
literals in source and never logged.

All backends expose::

    backend.chat(messages, system=None, max_tokens=512, temperature=None,
                 timeout=None, stop=None) -> str

with retry + exponential backoff and a per-call timeout.

Python 3.9 compatible (no ``X | Y`` annotations, no ``dict[str, ...]``).
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LLMError(RuntimeError):
    """Base class for backend failures."""


class LLMConfigError(LLMError):
    """Missing dependency or missing API key."""


class LLMCallError(LLMError):
    """A call failed after exhausting retries."""


Message = Dict[str, str]  # {"role": "user"|"assistant", "content": "..."}


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class LLMBackend(ABC):
    """Common retry / timing / bookkeeping wrapper around a chat model."""

    #: Subclasses may narrow this; used only for the default retry predicate.
    RETRYABLE_SUBSTRINGS = (
        "timeout", "timed out", "rate limit", "ratelimit", "429",
        "overloaded", "503", "502", "500", "529",
        "connection", "temporarily unavailable", "server error",
    )

    def __init__(
        self,
        model: str,
        max_retries: int = 4,
        base_delay: float = 0.5,
        max_delay: float = 8.0,
        timeout: float = 60.0,
        jitter: bool = True,
        seed: Optional[int] = None,
    ):
        self.model = model
        self.max_retries = int(max_retries)
        self.base_delay = float(base_delay)
        self.max_delay = float(max_delay)
        self.timeout = float(timeout)
        self.jitter = bool(jitter)
        self._rng = random.Random(seed if seed is not None else 0xC0DE)

        # Bookkeeping the harness reports on.
        self.calls = 0
        self.retries = 0
        self.failures = 0
        self.total_latency = 0.0
        self.latencies: List[float] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0

    # -- public API --------------------------------------------------------

    def chat(
        self,
        messages: Sequence[Message],
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> str:
        """Send a chat turn and return the assistant text.

        Retries transient failures with exponential backoff.  ``timeout`` is
        per attempt (seconds); the total wall clock can therefore reach
        ``timeout * (max_retries + 1)`` plus backoff.
        """
        msgs = [dict(m) for m in messages]
        effective_timeout = self.timeout if timeout is None else float(timeout)

        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            started = time.time()
            try:
                text = self._complete(
                    messages=msgs,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=effective_timeout,
                    stop=list(stop) if stop else None,
                )
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                last_exc = exc
                if attempt >= self.max_retries or not self.is_retryable(exc):
                    break
                self.retries += 1
                self._sleep(self._backoff(attempt))
                continue
            else:
                elapsed = time.time() - started
                self.calls += 1
                self.total_latency += elapsed
                self.latencies.append(elapsed)
                return text if text is not None else ""

        self.failures += 1
        raise LLMCallError(
            "%s call failed after %d attempt(s): %s"
            % (type(self).__name__, self.max_retries + 1, last_exc)
        )

    def is_retryable(self, exc: BaseException) -> bool:
        if isinstance(exc, (LLMConfigError, KeyboardInterrupt, SystemExit)):
            return False
        blob = ("%s %s" % (type(exc).__name__, exc)).lower()
        return any(s in blob for s in self.RETRYABLE_SUBSTRINGS)

    def stats(self) -> Dict[str, Any]:
        mean = self.total_latency / self.calls if self.calls else 0.0
        return {
            "backend": type(self).__name__,
            "model": self.model,
            "calls": self.calls,
            "retries": self.retries,
            "failures": self.failures,
            "mean_latency_s": mean,
            "total_latency_s": self.total_latency,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    # -- internals ---------------------------------------------------------

    def _backoff(self, attempt: int) -> float:
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        if self.jitter:
            delay *= 0.5 + self._rng.random()
        return delay

    def _sleep(self, seconds: float) -> None:  # overridable in tests
        time.sleep(seconds)

    @staticmethod
    def _require_env(var: str) -> str:
        value = os.environ.get(var)
        if not value:
            raise LLMConfigError(
                "environment variable %s is not set; export it before running "
                "(keys are never read from source or config files)" % var
            )
        return value

    @abstractmethod
    def _complete(
        self,
        messages: List[Message],
        system: Optional[str],
        max_tokens: int,
        temperature: Optional[float],
        timeout: float,
        stop: Optional[List[str]],
    ) -> str:
        """Perform exactly one provider call.  Raise on failure."""


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIBackend(LLMBackend):
    """OpenAI Chat Completions.  Requires ``pip install openai``."""

    ENV_VAR = "OPENAI_API_KEY"

    def __init__(self, model: str = "gpt-4o-2024-05-13", base_url: Optional[str] = None,
                 **kwargs: Any):
        super().__init__(model=model, **kwargs)
        self.base_url = base_url
        self._client = None  # created lazily so import/env errors surface late

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:  # pragma: no cover - env dependent
                raise LLMConfigError(
                    "the 'openai' package is required for OpenAIBackend "
                    "(pip install openai)"
                ) from exc
            kwargs = {"api_key": self._require_env(self.ENV_VAR), "max_retries": 0}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _complete(self, messages, system, max_tokens, temperature, timeout, stop):
        payload = list(messages)
        if system:
            payload = [{"role": "system", "content": system}] + payload

        kwargs = {
            "model": self.model,
            "messages": payload,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if stop:
            kwargs["stop"] = stop

        response = self._get_client().chat.completions.create(**kwargs)

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicBackend(LLMBackend):
    """Anthropic Messages API.  Requires ``pip install anthropic``.

    Note: current-generation Claude models reject non-default sampling
    parameters, so ``temperature`` is only forwarded when explicitly set.
    """

    ENV_VAR = "ANTHROPIC_API_KEY"

    def __init__(self, model: str = "claude-opus-5", **kwargs: Any):
        super().__init__(model=model, **kwargs)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic  # type: ignore
            except ImportError as exc:  # pragma: no cover - env dependent
                raise LLMConfigError(
                    "the 'anthropic' package is required for AnthropicBackend "
                    "(pip install anthropic)"
                ) from exc
            self._client = anthropic.Anthropic(
                api_key=self._require_env(self.ENV_VAR),
                max_retries=0,  # retries are handled by LLMBackend.chat
            )
        return self._client

    def _complete(self, messages, system, max_tokens, temperature, timeout, stop):
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": list(messages),
            "timeout": timeout,
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        if stop:
            kwargs["stop_sequences"] = stop

        response = self._get_client().messages.create(**kwargs)

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.prompt_tokens += getattr(usage, "input_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "output_tokens", 0) or 0

        parts = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts)


# ---------------------------------------------------------------------------
# Mock (offline, deterministic)
# ---------------------------------------------------------------------------

class MockBackend(LLMBackend):
    """Deterministic offline backend for tests and dry runs.

    Resolution order for each call:

    1. ``rules`` -- list of ``(regex, response)`` pairs matched against the
       concatenated prompt (system + all message contents).  First match wins.
       ``response`` may be a string or ``callable(prompt) -> str``.
    2. ``responses`` -- a scripted queue consumed in order.  When exhausted it
       either cycles (``cycle=True``, the default) or falls through to (3).
    3. ``default`` -- a string, a ``callable(prompt) -> str``, or ``None`` to
       use a deterministic hash-derived pick from :attr:`FALLBACKS`.

    ``fail_times`` makes the first N calls raise a retryable error, which is
    how the retry/backoff path is exercised without a network.
    """

    FALLBACKS = ("OK", "YES", "NO", "UNKNOWN", "PASS")

    def __init__(
        self,
        responses: Optional[Sequence[Any]] = None,
        rules: Optional[Sequence[Tuple[str, Any]]] = None,
        default: Optional[Any] = None,
        cycle: bool = True,
        fail_times: int = 0,
        latency: float = 0.0,
        model: str = "mock",
        **kwargs: Any,
    ):
        kwargs.setdefault("base_delay", 0.0)
        kwargs.setdefault("max_delay", 0.0)
        kwargs.setdefault("jitter", False)
        super().__init__(model=model, **kwargs)
        self.responses = list(responses or [])
        self.rules = [(re.compile(pat, re.I | re.S), out) for pat, out in (rules or [])]
        self.default = default
        self.cycle = cycle
        self.remaining_failures = int(fail_times)
        self.latency = float(latency)
        self.prompts: List[str] = []
        self._cursor = 0

    # Mock never really sleeps, so backoff does not slow the suite down.
    def _sleep(self, seconds: float) -> None:
        return None

    def is_retryable(self, exc: BaseException) -> bool:
        if isinstance(exc, LLMConfigError):
            return False
        return True

    @staticmethod
    def render_prompt(messages: Sequence[Message], system: Optional[str]) -> str:
        chunks = []
        if system:
            chunks.append("system: %s" % system)
        for m in messages:
            chunks.append("%s: %s" % (m.get("role", "user"), m.get("content", "")))
        return "\n".join(chunks)

    def _complete(self, messages, system, max_tokens, temperature, timeout, stop):
        prompt = self.render_prompt(messages, system)
        self.prompts.append(prompt)

        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise LLMCallError("mock transient failure (simulated timeout)")

        if self.latency:
            time.sleep(self.latency)

        for pattern, out in self.rules:
            if pattern.search(prompt):
                return self._render(out, prompt)

        if self.responses:
            if self._cursor < len(self.responses):
                out = self.responses[self._cursor]
                self._cursor += 1
                return self._render(out, prompt)
            if self.cycle:
                out = self.responses[self._cursor % len(self.responses)]
                self._cursor += 1
                return self._render(out, prompt)

        if self.default is not None:
            return self._render(self.default, prompt)

        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return self.FALLBACKS[int(digest[:8], 16) % len(self.FALLBACKS)]

    @staticmethod
    def _render(out: Any, prompt: str) -> str:
        if callable(out):
            return str(out(prompt))
        return str(out)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_BACKENDS = {
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
    "claude": AnthropicBackend,
    "mock": MockBackend,
}


def get_backend(name: str, **kwargs: Any) -> LLMBackend:
    """Instantiate a backend by short name ('openai' / 'anthropic' / 'mock')."""
    key = str(name).strip().lower()
    if key not in _BACKENDS:
        raise LLMConfigError(
            "unknown backend %r (choose from %s)" % (name, sorted(_BACKENDS))
        )
    return _BACKENDS[key](**kwargs)


def available_backends() -> List[str]:
    return sorted(_BACKENDS)
