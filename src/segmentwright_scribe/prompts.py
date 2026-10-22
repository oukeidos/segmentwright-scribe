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
