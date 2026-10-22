from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .presets import PRESETS
from .prompts import NO_SPEECH, PLAIN_PROMPTS, SUBTITLE_LINES_PROMPTS


class OpenRouterError(RuntimeError):
    """Raised when an OpenRouter transcription call fails."""


@dataclass(frozen=True)
class AsrRequest:
    request_id: str
    audio_path: Path
    model: str
    language: str
    timeout_sec: float
    preset: str | None = None


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
        self._models: list[dict[str, Any]] | None = None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def resolve_model(self, model: str, timeout_sec: float) -> dict[str, Any]:
        """Discover capabilities once per client; never infer them from a model name."""
        if self._models is None:
            for attempt in range(self.max_retries + 1):
                try:
                    response = self.client.get(
                        f"{self.base_url}/models",
                        params={"output_modalities": "all"},
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        timeout=timeout_sec,
                    )
                    response.raise_for_status()
                    body = response.json()
                    models = body.get("data") if isinstance(body, dict) else None
                    if not isinstance(models, list):
                        raise ValueError("Missing models data array")
                    self._models = [m for m in models if isinstance(m, dict)]
                    break
                except (httpx.HTTPError, ValueError) as exc:
                    retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                        exc.response.status_code in {429, 500, 502, 503, 504}
                    )
                    if not retryable or attempt == self.max_retries:
                        raise OpenRouterError(
                            f"Model discovery failed after {attempt + 1} attempt(s): {exc}"
                        ) from exc
        matches = [m for m in self._models or [] if m.get("id") == model]
        if len(matches) != 1:
            raise OpenRouterError(f"Model not uniquely listed in OpenRouter catalog: {model}")
        metadata = matches[0]
        architecture = metadata.get("architecture")
        if not isinstance(architecture, dict):
            raise OpenRouterError(f"Missing architecture metadata for {model}")
        inputs = architecture.get("input_modalities")
        outputs = architecture.get("output_modalities")
        if not isinstance(inputs, list) or not isinstance(outputs, list) or "audio" not in inputs:
            raise OpenRouterError(f"Model does not declare audio input support: {model}")
        if "transcription" in outputs:
            endpoint = "/audio/transcriptions"
        elif "text" in outputs:
            endpoint = "/chat/completions"
        else:
            raise OpenRouterError(f"Model has no supported transcription/text output: {model}")
        return {"endpoint": endpoint, "model": metadata}

    def request_profile(
        self, model: str, language: str, timeout_sec: float, preset: str | None = None
    ) -> str:
        route = self.resolve_model(model, timeout_sec)
        selected = validate_preset(preset, model, route["endpoint"]) if preset else None
        profile = {
            "endpoint": route["endpoint"],
            "prompt": chat_prompt(language, preset)
            if route["endpoint"] == "/chat/completions"
            else None,
            "version": 1,
        }
        if selected:
            profile.update(options=selected.options(), output_version=selected.output_version)
        return hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()

    def transcribe(self, request: AsrRequest) -> AsrResult:
        route = self.resolve_model(request.model, request.timeout_sec)
        if request.preset:
            validate_preset(request.preset, request.model, route["endpoint"])
        started = time.monotonic()
        last_error: Exception | None = None
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self.client.post(
                    f"{self.base_url}{route['endpoint']}",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=build_payload(request, endpoint=route["endpoint"]),
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
                if data.get("error"):
                    raise OpenRouterError(f"OpenRouter returned an error: {data['error']}")
                if route["endpoint"] == "/chat/completions":
                    choices = data.get("choices")
                    if (
                        not isinstance(choices, list)
                        or not choices
                        or not isinstance(choices[0], dict)
                    ):
                        raise OpenRouterError("Chat response has no completion choice")
                    if choices[0].get("finish_reason") != "stop":
                        raise OpenRouterError("Chat transcription incomplete or filtered")
                    message = choices[0].get("message")
                    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                        raise OpenRouterError("Chat response has no text content")
                    text = message["content"]
                    if not text.strip():
                        raise OpenRouterError("Chat response contains empty text")
                    if (
                        request.preset
                        and PRESETS[request.preset].output_mode == "json_string_lines"
                    ):
                        text = json_string_lines(text)
                    elif text.strip() == NO_SPEECH[request.language]:
                        text = ""
                else:
                    if request.preset:
                        mode = PRESETS[request.preset].output_mode
                        text = (
                            speaker_word_lines(data)
                            if mode == "speaker_words"
                            else segment_lines(data)
                        )
                    else:
                        text = extract_response_text(data)
                return AsrResult(
                    request_id=request.request_id,
                    text=text,
                    raw_response=data,
                    elapsed_sec=round(time.monotonic() - started, 3),
                    attempts=attempt,
                    status_code=response.status_code,
                )
            except (
                httpx.TimeoutException,
                httpx.TransportError,
                OpenRouterError,
                ValueError,
            ) as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                continue
        raise OpenRouterError(
            f"OpenRouter transcription failed after {attempts} attempt(s): {last_error}"
        ) from last_error


def build_payload(
    request: AsrRequest, *, endpoint: str = "/audio/transcriptions"
) -> dict[str, Any]:
    selected = validate_preset(request.preset, request.model, endpoint) if request.preset else None
    audio_bytes = request.audio_path.read_bytes()
    if endpoint == "/chat/completions":
        payload = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": chat_prompt(request.language, request.preset)},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Strictly follow the transcription instruction."},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(audio_bytes).decode("ascii"),
                                "format": "wav",
                            },
                        },
                    ],
                },
            ],
        }
        if selected:
            payload.update(selected.options())
        return payload
    if endpoint != "/audio/transcriptions":
        raise OpenRouterError(f"Unsupported endpoint: {endpoint}")
    payload = {
        "model": request.model,
        "input_audio": {
            "data": base64.b64encode(audio_bytes).decode("ascii"),
            "format": "wav",
        },
        "language": request.language,
    }

    if request.preset:
        payload.update(validate_preset(request.preset, request.model, endpoint).options())
    return payload


def validate_preset(name, model, endpoint):
    preset = PRESETS.get(name)
    if preset is None or preset.model != model:
        raise OpenRouterError(f"Preset/model mismatch: {name} / {model}")
    expected = (
        "/chat/completions"
        if preset.output_mode == "json_string_lines"
        else "/audio/transcriptions"
    )
    if endpoint != expected:
        raise OpenRouterError(f"Preset requires {expected} capabilities in server metadata")
    return preset


def chat_prompt(language: str, preset: str | None) -> str:
    prompts = (
        SUBTITLE_LINES_PROMPTS
        if preset and PRESETS[preset].output_mode == "json_string_lines"
        else PLAIN_PROMPTS
    )
    return prompts[language]


def json_string_lines(raw: str) -> str:
    """Validate the JSON contract and join lines without rewriting any string."""
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise OpenRouterError("Preset requires valid JSON string lines") from exc
    if not isinstance(data, dict) or set(data) != {"lines"} or not isinstance(data["lines"], list):
        raise OpenRouterError("Preset requires an object containing only a lines array")
    if any(
        not isinstance(line, str) or not line.strip() or "\n" in line or "\r" in line
        for line in data["lines"]
    ):
        raise OpenRouterError("Preset requires nonempty single-line strings")
    return "\n".join(data["lines"])


def speaker_word_lines(data: dict[str, Any]) -> str:
    """Insert only newlines; retain every original character and whitespace."""
    text, words = data.get("text"), data.get("words")
    if not isinstance(text, str) or not isinstance(words, list):
        raise OpenRouterError("Preset requires text and words in the response")
    positions = [i for i, char in enumerate(text) if not char.isspace()]
    source = "".join(text.split())
    cursor = 0
    previous = None
    boundaries = []
    for word in words:
        if not isinstance(word, dict) or not isinstance(word.get("word"), str):
            raise OpenRouterError("Preset received an invalid word")
        token = "".join(word["word"].split())
        if not token:
            continue
        speaker = word.get("speaker")
        if speaker is None:
            raise OpenRouterError("Preset requires a speaker label for every nonempty word")
        if source[cursor : cursor + len(token)] != token:
            raise OpenRouterError("Word text differs from the full transcript")
        if previous is not None and speaker != previous:
            boundaries.append(positions[cursor])
        cursor += len(token)
        previous = speaker
    if cursor != len(source):
        raise OpenRouterError("Words do not cover the full transcript")
    for position in reversed(boundaries):
        text = text[:position] + "\n" + text[position:]
    return text


def segment_lines(data: dict[str, Any]) -> str:
    segments = data.get("segments")
    full_text = data.get("text")
    if not isinstance(segments, list) or not isinstance(full_text, str):
        raise OpenRouterError("Preset requires text and segments in the response")
    lines = []
    for segment in segments:
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise OpenRouterError("Preset received an invalid segment")
        text = " ".join(segment["text"].split())
        if not text:
            continue
        if segment.get("speaker") is None:
            raise OpenRouterError("Preset requires a speaker label for every nonempty segment")
        lines.append(text)
    result = "\n".join(lines)
    if "".join(result.split()) != "".join(full_text.split()):
        raise OpenRouterError("Segment text differs from the full transcript")
    return result


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
