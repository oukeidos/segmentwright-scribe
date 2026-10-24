# Changelog

## [0.3.0] - 2026-10-24

- Added the optional beta `local` preset with isolated Granite and Qwen workers and `local install` / `local doctor` commands, while preserving existing cloud functionality.

## [0.2.1] - 2026-10-23

- Made intermediate files temporary by default with cleanup after each run and next-run cleanup of app-marked temporary folders older than 24 hours, while preserving explicitly selected artifact/cache paths and VAD models.

## [0.2.0] - 2026-10-22

- Added `--preset` and `--list-presets` with MAI-Transcribe 2, Scribe v2, and Qwen 3.8 Omni Flash presets for improved line breaks.

- Added automatic API routing from OpenRouter model metadata, with prompt-free ASR and Segmentwright plain prompts for multimodal LLMs.

## [0.1.0] - 2026-09-08

- Initial release.
