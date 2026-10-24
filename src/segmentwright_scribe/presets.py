"""Explicit model presets; API routes still come from server metadata."""

from copy import deepcopy
from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    model: str
    description: str
    output_version: str
    request_options: dict
    output_mode: str = "segments"

    def options(self) -> dict:
        return deepcopy(self.request_options)


PRESETS = {
    "local": Preset(
        model="local/granite-qwen",
        description="Offline Granite transcription with one Qwen repair pass (beta)",
        output_version="local-transcript-v1",
        request_options={},
        output_mode="local",
    ),
    "qwen-3.8-omni-flash": Preset(
        model="qwen/qwen3.8-omni-flash",
        description="Subtitle lines as JSON strings; preserve returned text and line breaks",
        output_mode="json_string_lines",
        output_version="subtitle-string-lines-v1",
        request_options={
            "response_format": {"type": "json_object"},
            "reasoning": {"enabled": False},
            "max_tokens": 8192,
            "provider": {"require_parameters": True},
        },
    ),
    "mai-transcribe-2": Preset(
        model="microsoft/mai-transcribe-2",
        description="Verbatim transcription with speaker-based line breaks",
        output_version="server-segment-lines-v1",
        request_options={
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
            "provider": {
                "options": {
                    "azure": {
                        "diarization": {"enabled": True},
                        "enhancedMode": {"modelOptions": {"transcribeStyle": "verbatim"}},
                    }
                }
            },
        },
    ),
    "scribe-v2": Preset(
        model="elevenlabs/scribe-v2",
        description="Preserve transcript text; add newlines at speaker changes",
        output_mode="speaker_words",
        output_version="speaker-word-newlines-v1",
        request_options={
            "response_format": "verbose_json",
            "timestamp_granularities": ["word"],
            "provider": {
                "options": {
                    "elevenlabs": {
                        "diarize": True,
                        "tag_audio_events": False,
                        "no_verbatim": False,
                    }
                }
            },
        },
    ),
}
