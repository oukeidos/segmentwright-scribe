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
        if request.method == "GET":
            return catalog("microsoft/mai-transcribe-2")
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
    assert seen_headers == ["Bearer secret", "Bearer secret"]


def test_openrouter_transcriber_retries_429(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"abc")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.method == "GET":
            return catalog("model")
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

    result = transcriber.transcribe(AsrRequest("chunk_0001", audio, "model", "en", 10.0))

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


def catalog(model="new/model", inputs=None, outputs=None):
    return httpx.Response(
        200,
        json={
            "data": [
                {
                    "id": model,
                    "architecture": {
                        "input_modalities": inputs if inputs is not None else ["audio"],
                        "output_modalities": outputs if outputs is not None else ["transcription"],
                    },
                }
            ]
        },
    )


@pytest.mark.parametrize("language", ["en", "ja"])
@pytest.mark.parametrize(
    "outputs,endpoint",
    [
        (["transcription"], "/audio/transcriptions"),
        (["text"], "/chat/completions"),
        (["transcription", "text"], "/audio/transcriptions"),
    ],
)
def test_discovery_routes_unlisted_names_and_reuses_catalog(tmp_path, language, outputs, endpoint):
    from segmentwright_scribe.prompts import NO_SPEECH, PLAIN_PROMPTS

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    calls = []

    def handler(request):
        calls.append(request.method)
        if request.method == "GET":
            assert request.url.params["output_modalities"] == "all"
            return catalog(outputs=outputs)
        assert request.url.path == "/api/v1" + endpoint
        body = json.loads(request.content)
        if endpoint == "/audio/transcriptions":
            assert set(body) == {"model", "input_audio", "language"}
            assert body["language"] == language
            return httpx.Response(200, json={"text": "hello"})
        assert "language" not in body
        assert body["messages"][0]["content"] == PLAIN_PROMPTS[language]
        audio_data = body["messages"][1]["content"][1]["input_audio"]["data"]
        assert base64.b64decode(audio_data) == b"audio"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": NO_SPEECH[language]},
                    }
                ]
            },
        )

    transcriber = OpenRouterTranscriber(
        api_key="secret",
        base_url="https://test/api/v1",
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    request = AsrRequest("1", audio, "new/model", language, 10)
    for _ in range(2):
        result = transcriber.transcribe(request)
        assert result.text == ("hello" if endpoint == "/audio/transcriptions" else "")
    assert calls == ["GET", "POST", "POST"]


@pytest.mark.parametrize(
    "response",
    [
        catalog(inputs=["text"]),
        catalog(outputs=["speech"]),
        httpx.Response(200, json={"data": []}),
        httpx.Response(200, json={"data": [{"id": "new/model"}]}),
        httpx.Response(200, json={"unexpected": []}),
        httpx.Response(401, json={"error": "unauthorized"}),
    ],
)
def test_discovery_failure_never_uploads_audio(tmp_path, response):
    calls = []

    def handler(request):
        calls.append(request.method)
        return response

    transcriber = OpenRouterTranscriber(
        api_key="secret",
        base_url="https://test/api/v1",
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(OpenRouterError):
        transcriber.transcribe(AsrRequest("1", tmp_path / "absent.wav", "new/model", "en", 10))
    assert calls == ["GET"]


@pytest.mark.parametrize(
    "choice",
    [
        {"finish_reason": "length", "message": {"content": "partial"}},
        {"finish_reason": "content_filter", "message": {"content": "partial"}},
        {"finish_reason": "stop", "message": {"content": "", "reasoning": "hidden"}},
        {"finish_reason": "stop", "message": {"content": None}},
    ],
)
def test_invalid_chat_is_not_success(tmp_path, choice):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")

    def handler(request):
        return (
            catalog(outputs=["text"])
            if request.method == "GET"
            else httpx.Response(200, json={"choices": [choice]})
        )

    transcriber = OpenRouterTranscriber(
        api_key="secret",
        base_url="https://test/api/v1",
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(OpenRouterError):
        transcriber.transcribe(AsrRequest("1", audio, "new/model", "en", 10))


def test_request_profile_changes_with_route_and_prompt(monkeypatch):
    from segmentwright_scribe.prompts import PLAIN_PROMPTS

    def make(outputs):
        return OpenRouterTranscriber(
            api_key="secret",
            base_url="https://test/api/v1",
            max_retries=0,
            client=httpx.Client(
                transport=httpx.MockTransport(lambda request: catalog(outputs=outputs))
            ),
        )

    asr = make(["transcription"])
    chat = make(["text"])
    first = chat.request_profile("new/model", "en", 10)
    assert first != asr.request_profile("new/model", "en", 10)
    monkeypatch.setitem(PLAIN_PROMPTS, "en", "changed prompt")
    assert first != chat.request_profile("new/model", "en", 10)


@pytest.mark.parametrize("lang", ["en", "ja"])
def test_mai_preset_contract_and_segment_lines(tmp_path, lang):
    model = "microsoft/mai-transcribe-2"
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"audio")

    def handler(request):
        if request.method == "GET":
            return catalog(model)
        body = json.loads(request.content)
        assert request.url.path.endswith("/audio/transcriptions")
        assert "prompt" not in body and "messages" not in body
        assert body["response_format"] == "verbose_json"
        assert body["timestamp_granularities"] == ["segment"]
        opts = body["provider"]["options"]["azure"]
        assert opts["diarization"]["enabled"] is True
        assert opts["enhancedMode"]["modelOptions"]["transcribeStyle"] == "verbatim"
        return httpx.Response(
            200,
            json={
                "text": "Hello. Yes. Again.",
                "segments": [
                    {"text": "Hello.", "speaker": 0, "start": 10},
                    {"text": "Yes.", "speaker": 1, "start": 20},
                    {"text": "Again.", "speaker": 1, "start": 30},
                ],
            },
        )

    t = OpenRouterTranscriber(
        api_key="secret",
        base_url="https://test/api/v1",
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert t.request_profile(model, lang, 10) != t.request_profile(
        model, lang, 10, preset="mai-transcribe-2"
    )
    result = t.transcribe(AsrRequest("1", audio, model, lang, 10, "mai-transcribe-2"))
    assert result.text == "Hello.\nYes.\nAgain."


@pytest.mark.parametrize(
    "data",
    [
        {"text": "hello"},
        {"text": "hello", "segments": []},
        {"text": "hello", "segments": [{"text": "hello"}]},
        {"text": "hello", "segments": [{"text": "other", "speaker": 0}]},
    ],
)
def test_preset_does_not_silently_fall_back(data):
    from segmentwright_scribe.openrouter import segment_lines

    with pytest.raises(OpenRouterError):
        segment_lines(data)


def test_preset_accepts_empty_speech():
    from segmentwright_scribe.openrouter import segment_lines

    assert segment_lines({"text": "", "segments": []}) == ""


@pytest.mark.parametrize(
    "text,words,expected",
    [
        ("Hello,  world!", [("Hello,", 0), ("world!", 1)], "Hello,  \nworld!"),
        ("はい。こんにちは！", [("はい。", 0), ("こんにちは！", 1)], "はい。\nこんにちは！"),
        ("A\n B\tC", [("A", 0), ("B", 0), ("C", 1)], "A\n B\t\nC"),
        (" ", [], " "),
        ("", [], ""),
    ],
)
def test_scribe_preserves_original_characters(text, words, expected):
    from segmentwright_scribe.openrouter import speaker_word_lines

    assert (
        speaker_word_lines(
            {"text": text, "words": [{"word": word, "speaker": speaker} for word, speaker in words]}
        )
        == expected
    )


@pytest.mark.parametrize(
    "data",
    [
        {"text": "hi"},
        {"text": "hi", "words": []},
        {"text": "hi", "words": [{"word": "hi"}]},
        {"text": "hi", "words": [{"word": "bye", "speaker": 0}]},
        {"text": "hi", "words": [{"word": "hi", "speaker": 0}, {"word": "extra", "speaker": 1}]},
    ],
)
def test_scribe_rejects_unmappable_words(data):
    from segmentwright_scribe.openrouter import speaker_word_lines

    with pytest.raises(OpenRouterError):
        speaker_word_lines(data)


def test_scribe_preset_http_contract(tmp_path):
    model = "elevenlabs/scribe-v2"
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"audio")

    def handler(request):
        if request.method == "GET":
            return catalog(model)
        body = json.loads(request.content)
        assert body["timestamp_granularities"] == ["word"]
        assert body["response_format"] == "verbose_json"
        assert body["provider"] == {
            "options": {
                "elevenlabs": {
                    "diarize": True,
                    "tag_audio_events": False,
                    "no_verbatim": False,
                }
            }
        }
        assert "prompt" not in body
        return httpx.Response(
            200,
            json={
                "text": "A B",
                "words": [
                    {"word": "A", "speaker": 0},
                    {"word": "B", "speaker": 1},
                ],
            },
        )

    t = OpenRouterTranscriber(
        api_key="secret",
        base_url="https://test/api/v1",
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert t.request_profile(model, "en", 10) != t.request_profile(
        model, "en", 10, preset="scribe-v2"
    )
    assert t.transcribe(AsrRequest("1", audio, model, "en", 10, "scribe-v2")).text == "A \nB"


@pytest.mark.parametrize("lang", ["en", "ja"])
@pytest.mark.parametrize("lines", [[], ["  Hello,  world! ", " はい。\t", "Again…"]])
def test_qwen_preset_preserves_json_lines_and_options(tmp_path, lang, lines, monkeypatch):
    from segmentwright_scribe.prompts import SUBTITLE_LINES_PROMPTS

    model = "qwen/qwen3.8-omni-flash"
    preset = "qwen-3.8-omni-flash"
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"audio")

    def handler(request):
        if request.method == "GET":
            return catalog(model, outputs=["text"])
        assert request.url.path.endswith("/chat/completions")
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        assert body["reasoning"] == {"enabled": False}
        assert body["max_tokens"] == 8192
        assert body["provider"] == {"require_parameters": True}
        assert body["messages"][0]["content"] == SUBTITLE_LINES_PROMPTS[lang]
        assert "array of strings" in body["messages"][0]["content"]
        assert body["messages"][1]["content"][0]["text"] == (
            "Strictly follow the transcription instruction."
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps({"lines": lines}, ensure_ascii=False)},
                    }
                ]
            },
        )

    t = OpenRouterTranscriber(
        api_key="secret",
        base_url="https://test/api/v1",
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    profile = t.request_profile(model, lang, 10, preset)
    assert profile != t.request_profile(model, lang, 10)
    assert t.transcribe(AsrRequest("1", audio, model, lang, 10, preset)).text == "\n".join(lines)
    monkeypatch.setitem(SUBTITLE_LINES_PROMPTS, lang, "changed subtitle prompt")
    assert profile != t.request_profile(model, lang, 10, preset)


@pytest.mark.parametrize(
    "raw",
    [
        "not JSON",
        "[]",
        '{"lines": "text"}',
        '{"lines":[{"text":"hi"}]}',
        '{"lines":[1]}',
        '{"lines":[null]}',
        '{"lines":[""]}',
        '{"lines":[" "]}',
        '{"lines":["hi\\nthere"]}',
        '{"lines":["hi"],"extra":""}',
    ],
)
def test_qwen_rejects_invalid_contract_without_rewriting(raw):
    from segmentwright_scribe.openrouter import json_string_lines

    with pytest.raises(OpenRouterError):
        json_string_lines(raw)


def test_qwen_metadata_route_mismatch_never_uploads(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.method)
        return catalog("qwen/qwen3.8-omni-flash", outputs=["transcription"])

    t = OpenRouterTranscriber(
        api_key="secret",
        base_url="https://test/api/v1",
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(OpenRouterError, match="capabilities"):
        t.transcribe(
            AsrRequest(
                "1",
                tmp_path / "absent.wav",
                "qwen/qwen3.8-omni-flash",
                "en",
                10,
                "qwen-3.8-omni-flash",
            )
        )
    assert calls == ["GET"]
