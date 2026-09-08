# Segmentwright Scribe

Segmentwright Scribe is a companion tool for Segmentwright: it transcribes audio
or video into JSON that Segmentwright uses to create subtitles.

Scribe splits audio with FireRedVAD and sends the chunks to OpenRouter, using
`microsoft/mai-transcribe-2` by default. It runs independently during transcription;
then you pass the resulting JSON to Segmentwright with `--transcript`.

## Setup

You need:

- `uv` and Python 3.12 (`uv` can manage Python for you).
- `ffmpeg` and `ffprobe` on `PATH`.
- An OpenRouter API key and network access for transcription and initial model downloads.
- A separate Segmentwright installation for the subtitle step.

Run these commands from the Scribe repository directory:

```bash
uv sync
uv run segmentwright-scribe --help
uv run segmentwright-scribe --version
```

On Ubuntu, install the media tools with `sudo apt install ffmpeg`.

Set your API key for the current shell:

```bash
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

For persistent configuration, create `~/.segmentwright-scribe/.env` containing:

```dotenv
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Keep the file private with `chmod 600 ~/.segmentwright-scribe/.env`. The shell
variable takes priority over the file. Use `--env-file` to choose another file or
`--openrouter-api-key-env` to change the variable name. API key values are not
accepted as CLI arguments.

## Transcribe

Pass an input media file and an output `.json` path. Japanese is the default:

```bash
uv run segmentwright-scribe input.mp4 transcript_chunks.json
```

For English:

```bash
uv run segmentwright-scribe input.webm transcript_chunks.json --language en
```

Supported inputs: `.mp4`, `.mkv`, `.mov`, `.webm`, `.mp3`, `.m4a`, and `.wav`.
Add `--verbose` for progress details and `--overwrite` to replace an existing output.

To keep diagnostics and cache in predictable locations:

```bash
uv run segmentwright-scribe input.mp4 runs/example/transcript_chunks.json \
  --artifacts runs/example/artifacts \
  --cache-dir runs/example/cache \
  --verbose
```

Cache is enabled by default. Reuse the same `--cache-dir` across runs to reuse
cached transcripts, or add `--no-cache` to request fresh transcriptions.
Use `--model MODEL_ID` to select another audio transcription model on OpenRouter.

## Create Subtitles with Segmentwright

Pass the JSON and the **same media file** to your Segmentwright installation.
Replace the paths below with your own:

```bash
uv run --project /path/to/segmentwright \
  segmentwright /path/to/input.mp4 /path/to/output.srt \
  --transcript /path/to/transcript_chunks.json \
  --language ja \
  --preset cheaper-candidate \
  --segmentation resegmentation \
  --resegmentation-model gemma4:12b \
  --resegmentation-prompt medium
```

Use `--language en` for English. The preset and resegmentation options are an
example; choose settings and a model available in your Segmentwright installation.
Add `--overwrite` when replacing an existing subtitle file.

## Local Voice Detection

FireRedVAD finds speech and plans chunks locally before OpenRouter transcribes them.
The default target and maximum chunk lengths are both 180 seconds.

Model files are downloaded automatically when the model directory is absent:

```text
~/.cache/segmentwright-scribe/FireRedVAD/VAD
```

Use `--firered-vad-model-dir DIR` for an existing model directory, or
`--no-firered-vad-auto-download` to disable downloads. The directory must contain
`model.pth.tar` and `cmvn.ark`.

`--device auto`, `cpu`, or `cuda` selects the device for FireRedVAD only.
Transcription runs remotely on OpenRouter.

## Outputs

The requested output uses the `segmentwright.transcript_chunks.v1` schema.
Diagnostics go to `artifacts/<input-stem>-<timestamp>` unless you set `--artifacts`:

| File or directory | Contents |
| --- | --- |
| `run_config.json`, `media.json` | Run settings and media information. |
| `vad.json`, `chunk_manifest.json` | Speech regions and planned chunks. |
| `source.wav`, `chunks/` | Normalized audio and audio chunks. |
| `openrouter_calls/` | Request metadata, responses, available usage, timing, retries, and empty-transcript status. |
| `transcript_chunks.json`, `run_summary.json` | Transcript copy and run summary. |
| `errors.json` | Pipeline failure details, when a run fails. |
| `cache/` | Cached transcripts, unless `--cache-dir` points elsewhere. |

The CLI rejects missing or unsupported inputs, output paths without `.json`,
and existing outputs unless `--overwrite` is set. Input and output must differ;
`--artifacts` cannot equal the output path. The cache directory cannot equal,
contain, or be inside the output path.

## CLI Options

Usage: `segmentwright-scribe INPUT OUTPUT [OPTIONS]`.
`--help` and `--version` work without inputs or an API key.

| Flag | Default | Description |
| --- | --- | --- |
| `--version` | — | Print the installed package version and exit. No inputs or API key required. |
| `--language {en,ja}` | `ja` | Language hint sent to the ASR model. |
| `--model MODEL` | `microsoft/mai-transcribe-2` | OpenRouter ASR model ID. |
| `--artifacts ARTIFACTS` | `artifacts/<input-stem>-<timestamp>` | Directory for run artifacts. |
| `--env-file ENV_FILE` | `~/.segmentwright-scribe/.env` | Dotenv file used after process environment lookup. |
| `--openrouter-api-key-env NAME` | `OPENROUTER_API_KEY` | Environment variable name containing the OpenRouter API key. |
| `--openrouter-base-url URL` | `https://openrouter.ai/api/v1` | OpenRouter API base URL. |
| `--target-chunk-sec SECONDS` | `180.0` | Preferred chunk duration. |
| `--max-chunk-sec SECONDS` | `180.0` | Hard maximum chunk duration before forced splitting. |
| `--min-retry-split-sec SECONDS` | `30.0` | Minimum duration considered for fallback chunk splitting. |
| `--firered-vad-model-dir DIR` | `~/.cache/segmentwright-scribe/FireRedVAD/VAD` | FireRedVAD model directory. |
| `--no-firered-vad-auto-download` | `False` | Fail if FireRedVAD files are missing. |
| `--device {auto,cuda,cpu}` | `auto` | Device used by FireRedVAD. |
| `--timeout-sec SECONDS` | `None` | Per-request OpenRouter timeout. If omitted, the timeout is computed per chunk. |
| `--max-request-retries N` | `3` | Retry count for transient OpenRouter failures. |
| `--cache-dir DIR` | `<artifacts>/cache` | Transcript cache directory. |
| `--no-cache` | `False` | Disable transcript cache reads and writes. |
| `--overwrite` | `False` | Allow replacing an existing `OUTPUT` file. |
| `--verbose` | `False` | Print progress details to stderr. |

## Troubleshooting

| Problem | Fix |
| --- | --- |
| API key not found | Set `OPENROUTER_API_KEY` in the shell or `.env` file. If you changed the variable name, use that name in both places. |
| `ffmpeg` or `ffprobe` missing | Install FFmpeg and check that both commands are on `PATH`. |
| Output already exists | Choose a new path or add `--overwrite`. |
| Unsupported input | Convert to one of the supported formats listed above. |
| FireRedVAD download fails | Check network access or supply a complete local model directory. For a partial download, remove the incomplete directory and retry. |
| `uv` cache permission error | Prefix the command with `UV_CACHE_DIR=/tmp/segmentwright-scribe-uv-cache`. |

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check
```

The release version is defined in `pyproject.toml` (`project.version`).
Python's `segmentwright_scribe.__version__` and CLI `--version` read the installed
package metadata. After a version change, run `uv lock` and `uv sync --extra dev`;
leave generated lock and package metadata files to those tools.

See [CHANGELOG.md](CHANGELOG.md) for releases and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party components and licenses.
