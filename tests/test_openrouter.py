from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest

from segmentwright_scribe.openrouter import (
    AsrRequest,
    OpenRouterError,
    OpenRouterTranscriber,
    build_payload,
)


def test_build_payload_embeds_base64_wav(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"abc")
    payload = build_payload(
        AsrRequest(
            request_id="chunk_0001",
            audio_path=audio,
            model="m",
            language="ja",
            timeout_sec=10.0,
        )
    )

    assert payload["input_audio"]["format"] == "wav"
    assert base64.b64decode(payload["input_audio"]["data"]) == b"abc"


def test_openrouter_transcriber_success_uses_authorization_header(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"abc")
    seen_headers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers["authorization"])
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "microsoft/mai-transcribe-2"
        return httpx.Response(200, json={"text": "hello"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    transcriber = OpenRouterTranscriber(
        api_key="secret",
        base_url="https://openrouter.test/api/v1",
        max_retries=0,
        client=client,
    )

    result = transcriber.transcribe(
        AsrRequest(
            request_id="chunk_0001",
            audio_path=audio,
            model="microsoft/mai-transcribe-2",
            language="en",
            timeout_sec=10.0,
        )
    )

    assert result.text == "hello"
    assert seen_headers == ["Bearer secret"]


def test_openrouter_transcriber_retries_429(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"abc")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json={"text": "ok"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    transcriber = OpenRouterTranscriber(
        api_key="secret",
        base_url="https://openrouter.test/api/v1",
        max_retries=1,
        client=client,
    )

    result = transcriber.transcribe(
        AsrRequest("chunk_0001", audio, "model", "en", 10.0)
    )

    assert result.text == "ok"
    assert result.attempts == 2


def test_openrouter_transcriber_raises_after_retries(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"abc")
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(503, text="down"))
    )
    transcriber = OpenRouterTranscriber(
        api_key="secret",
        base_url="https://openrouter.test/api/v1",
        max_retries=1,
        client=client,
    )

    with pytest.raises(OpenRouterError, match="after 2 attempt"):
        transcriber.transcribe(AsrRequest("chunk_0001", audio, "model", "en", 10.0))
