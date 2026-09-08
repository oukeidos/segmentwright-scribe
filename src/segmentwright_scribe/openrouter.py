from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


class OpenRouterError(RuntimeError):
    """Raised when an OpenRouter transcription call fails."""


@dataclass(frozen=True)
class AsrRequest:
    request_id: str
    audio_path: Path
    model: str
    language: str
    timeout_sec: float


@dataclass(frozen=True)
class AsrResult:
    request_id: str
    text: str
    raw_response: dict[str, Any]
    elapsed_sec: float
    attempts: int
    status_code: int

    @property
    def empty_text(self) -> bool:
        return self.text.strip() == ""


class OpenRouterTranscriber:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        max_retries: int,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.client = client or httpx.Client()
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def transcribe(self, request: AsrRequest) -> AsrResult:
        started = time.monotonic()
        last_error: Exception | None = None
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self.client.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=build_payload(request),
                    timeout=request.timeout_sec,
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < attempts:
                    continue
                if response.status_code >= 400:
                    raise OpenRouterError(
                        f"OpenRouter transcription failed with HTTP {response.status_code}: "
                        f"{response.text[:500]}"
                    )
                data = response.json()
                if not isinstance(data, dict):
                    raise OpenRouterError("OpenRouter response must be a JSON object")
                text = extract_response_text(data)
                return AsrResult(
                    request_id=request.request_id,
                    text=text,
                    raw_response=data,
                    elapsed_sec=round(time.monotonic() - started, 3),
                    attempts=attempt,
                    status_code=response.status_code,
                )
            except (httpx.TimeoutException, httpx.TransportError, OpenRouterError) as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                continue
        raise OpenRouterError(
            f"OpenRouter transcription failed after {attempts} attempt(s): {last_error}"
        ) from last_error


def build_payload(request: AsrRequest) -> dict[str, Any]:
    audio_bytes = request.audio_path.read_bytes()
    return {
        "model": request.model,
        "input_audio": {
            "data": base64.b64encode(audio_bytes).decode("ascii"),
            "format": "wav",
        },
        "language": request.language,
    }


def extract_response_text(data: dict[str, Any]) -> str:
    for key in ("text", "transcript", "content"):
        value = data.get(key)
        if isinstance(value, str):
            return value

    choices = data.get("choices")
    if isinstance(choices, list):
        chunks: list[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            text = choice.get("text")
            if isinstance(text, str):
                chunks.append(text)
                continue
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                chunks.append(message["content"])
        if chunks:
            return "".join(chunks)

    return ""
