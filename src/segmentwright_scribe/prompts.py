"""Plain transcript prompts copied from Segmentwright, without a runtime dependency.

Source: segmentwright/prompts.py at 0d181fd5945182e2c17ab60735dec3af2e8e9e58.
"""

PLAIN_PROMPTS = {
    "en": (
        "Transcribe all spoken English, including brief narration, voice-over, and bridge "
        "speech between dialogue turns. Output only the English transcription. Do not add "
        "timestamps, translations, explanations, markdown, speaker labels, or labels. If "
        "there is no spoken English to transcribe, output exactly this single line: "
        "((NO_SPOKEN_ENGLISH_TO_TRANSCRIBE)). Do not output any other text in that "
        "no-speech case."
    ),
    "ja": (
        "Transcribe all spoken Japanese, including brief narration, voice-over, and "
        "bridge speech between dialogue turns. Output only the Japanese transcription. Do "
        "not add timestamps, translations, explanations, markdown, speaker labels, or "
        "labels. If there is no spoken Japanese to transcribe, output exactly this single "
        "line: ((NO_SPOKEN_JAPANESE_TO_TRANSCRIBE)). Do not output any other text in that "
        "no-speech case."
    ),
}

NO_SPEECH = {
    "en": "((NO_SPOKEN_ENGLISH_TO_TRANSCRIBE))",
    "ja": "((NO_SPOKEN_JAPANESE_TO_TRANSCRIBE))",
}

# Tested subtitle-lines prompt with a direct string-array JSON contract.
SUBTITLE_LINES_PROMPTS = {
    "ja": (
        "Transcribe all spoken Japanese into subtitle-style transcript lines, including brief "
        "narration, voice-over, and bridge lines between dialogue turns. Start a new line "
        "when the speaker changes or may have changed. Also split long utterances more "
        "aggressively at Japanese punctuation and natural clause boundaries, especially after "
        "。, 、, ？, ！, or clear phrase endings. Keep each line short enough for video subtitles "
        'while preserving the spoken wording. Return only JSON with a "lines" array of '
        'strings, for example {"lines":["one transcript line","another transcript line"]}. '
        "Each string must contain exactly one transcript line. Do not add timestamps, "
        "translations, explanations, markdown, speaker labels, or extra fields. If there is "
        'no spoken Japanese, return exactly {"lines":[]}.'
    ),
    "en": (
        "Transcribe all spoken English into subtitle-style transcript lines, including brief "
        "narration, voice-over, and bridge speech between dialogue turns. Start a new line "
        "when the speaker changes or may have changed. Also split long utterances at English "
        "punctuation and natural clause boundaries, especially after periods, commas, "
        "question marks, exclamation marks, or clear phrase endings. Keep each line short "
        "enough for video subtitles while preserving the spoken wording. Return only JSON "
        'with a "lines" array of strings, for example {"lines":["one transcript '
        'line","another transcript line"]}. Each string must contain exactly one transcript '
        "line. Do not add timestamps, translations, explanations, markdown, speaker labels, "
        'or extra fields. If there is no spoken English, return exactly {"lines":[]}.'
    ),
}
