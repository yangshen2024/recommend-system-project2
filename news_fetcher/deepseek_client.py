"""
DeepSeek API Client

Encapsulates authentication, retries, rate limiting, and structured
request/response handling for the DeepSeek Chat Completions API.
"""

import json
import logging
import time
import threading
from typing import Any, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from news_fetcher.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_CHAT_MODEL,
    DEEPSEEK_MAX_RETRIES,
    DEEPSEEK_RATE_LIMIT_RPM,
    DEEPSEEK_TIMEOUT_SEC,
)

logger = logging.getLogger("deepseek")


# ── Rate Limiter (token bucket) ────────────────────────────
class RateLimiter:
    """Simple sliding-window rate limiter for API calls."""

    def __init__(self, max_calls: int, period_sec: float = 60.0):
        self.max_calls = max_calls
        self.period = period_sec
        self.calls: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a token is available; return wait time in seconds."""
        with self._lock:
            now = time.monotonic()
            # Purge expired entries
            self.calls = [t for t in self.calls if now - t < self.period]
            if len(self.calls) >= self.max_calls:
                wait = self.calls[0] + self.period - now
                if wait > 0:
                    time.sleep(wait)
                    now = time.monotonic()
                    self.calls = [t for t in self.calls if now - t < self.period]
            self.calls.append(now)
            return 0.0


# ── DeepSeek Client ────────────────────────────────────────
class DeepSeekClient:
    """Authenticated client for DeepSeek Chat API with retry & rate limiting."""

    CHAT_URL = f"{DEEPSEEK_BASE_URL}/v1/chat/completions"

    def __init__(self) -> None:
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY environment variable is not set.")

        self._api_key = DEEPSEEK_API_KEY
        self._httpx_client = httpx.Client(
            timeout=httpx.Timeout(DEEPSEEK_TIMEOUT_SEC),
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
        )
        self._rate_limiter = RateLimiter(max_calls=DEEPSEEK_RATE_LIMIT_RPM)

    # ── helpers ────────────────────────────────────────────
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # ── core chat interface ─────────────────────────────────
    @retry(
        retry=retry_if_exception_type(
            (httpx.HTTPStatusError, httpx.RemoteProtocolError, httpx.ReadTimeout)
        ),
        stop=stop_after_attempt(DEEPSEEK_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        response_format: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Send a chat completion request and return the parsed JSON body."""

        self._rate_limiter.acquire()

        payload: dict[str, Any] = {
            "model": DEEPSEEK_CHAT_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            # DeepSeek uses the OpenAI-compatible response_format
            payload["response_format"] = response_format

        logger.debug("Sending chat request (%d messages, %d tokens)", len(messages), max_tokens)

        resp = self._httpx_client.post(
            self.CHAT_URL,
            headers=self._headers(),
            json=payload,
        )

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After", "5")
            logger.warning("Rate limited by DeepSeek; waiting %ss", retry_after)
            time.sleep(float(retry_after))
            raise httpx.HTTPStatusError(
                f"429 Too Many Requests", request=resp.request, response=resp
            )

        resp.raise_for_status()
        return resp.json()

    # ── convenience: simple text completion ──────────────────
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """Single-turn completion returning the assistant's text response."""
        body = self.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            logger.error("Unexpected response shape: %s", json.dumps(body)[:500])
            raise ValueError(f"Could not extract content from DeepSeek response: {exc}") from exc

    # ── structured JSON extraction ───────────────────────────
    def extract_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """
        Request JSON output. Includes retry fallback in case the model
        returns markdown-wrapped JSON.
        """
        raw = self.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,  # low temperature for structured extraction
            max_tokens=max_tokens,
        )
        return self._parse_json_response(raw)

    @staticmethod
    def _parse_json_response(raw: str) -> dict[str, Any]:
        """Robust JSON extraction from model output (handles markdown fences)."""
        text = raw.strip()

        # Remove markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Drop first line (e.g. ```json) and last line (```)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("JSON parse failed; attempting repair on: %s", text[:300])
            # Last resort: try to find JSON object boundaries
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Failed to parse DeepSeek response as JSON: {text[:500]}")

    def close(self) -> None:
        self._httpx_client.close()
