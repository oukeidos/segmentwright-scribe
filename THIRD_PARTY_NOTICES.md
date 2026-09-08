# Third-Party Notices

Segmentwright Scribe uses the third-party projects listed below. Their code,
model files, names, and trademarks belong to their respective owners.

This document identifies direct dependencies, development tools, and external
components. Dependency declarations are maintained in `pyproject.toml`, and
resolved package versions are recorded in `uv.lock`. License identifiers below
were checked against the installed distributions on September 8, 2026.

## Python Runtime Dependencies

| Package | Project | Declared license | Purpose |
| --- | --- | --- | --- |
| `fireredvad` | [FireRedTeam/FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) | Apache-2.0 | Voice activity detection. |
| `huggingface-hub` | [Hugging Face Hub](https://github.com/huggingface/huggingface_hub) | Apache-2.0 | Download FireRedVAD model files. |
| `httpx` | [HTTPX](https://github.com/encode/httpx) | BSD-3-Clause | HTTP requests to OpenRouter. |
| `pydantic` | [Pydantic](https://github.com/pydantic/pydantic) | MIT | Transcript schema validation. |
| `python-dotenv` | [python-dotenv](https://github.com/theskumar/python-dotenv) | BSD-3-Clause | Read environment configuration files. |
| `torch` | [PyTorch](https://github.com/pytorch/pytorch) | BSD-3-Clause, with additional bundled-component licenses | Run the local VAD model. |

Copyright acknowledgments from the installed license files include:

- HTTPX: Copyright © 2019, Encode OSS Ltd. All rights reserved.
- Pydantic: Copyright (c) 2017 to present Pydantic Services Inc. and individual contributors.
- python-dotenv: Copyright (c) 2014, Saurabh Kumar (python-dotenv), 2013,
  Ted Tieken (django-dotenv-rw), 2013, Jacob Kaplan-Moss (django-dotenv).
- PyTorch: the PyTorch and Caffe2 contributors and the additional copyright
  holders identified in the distribution's `LICENSE` and `NOTICE` files.

## Build and Development Tools

| Package | Project | Declared license | Purpose |
| --- | --- | --- | --- |
| `setuptools` | [Setuptools](https://github.com/pypa/setuptools) | MIT | Build backend. |
| `pytest` | [pytest](https://github.com/pytest-dev/pytest) | MIT | Optional test runner. |
| `ruff` | [Ruff](https://github.com/astral-sh/ruff) | MIT | Optional linter. |

## FireRedVAD Model Files

The application can download `VAD/model.pth.tar` and `VAD/cmvn.ark` from
`FireRedTeam/FireRedVAD`. The upstream
[model card](https://huggingface.co/FireRedTeam/FireRedVAD/blob/main/README.md)
declares Apache-2.0. These files are downloaded separately into the model cache;
they are not included in this source repository.

## FFmpeg and ffprobe

The application invokes separately installed `ffmpeg` and `ffprobe` executables
to inspect and process media. This source repository does not include their
binaries. FFmpeg is licensed under LGPL-2.1-or-later, with optional GPL components
that can change the license of a particular build. See the official
[FFmpeg license information](https://ffmpeg.org/legal.html) and the license
information supplied with the installed build.

## Hosted Transcription

The application calls OpenRouter to access the configured ASR model. OpenRouter
and the model provider operate external services; their server code and ASR
model weights are not distributed with this project. Service access is governed
by the applicable provider terms.

## Complete License Texts and Transitive Dependencies

The tables above are an inventory of direct dependencies, not a complete license
bundle for an installed environment. Python distributions include their own
license and notice files, typically under `<package>-<version>.dist-info/licenses/`.
PyTorch in particular includes extensive third-party license texts and a separate
`NOTICE`. Transitive dependencies and platform-specific binary components carry
their own notices as well.

When distributing dependency packages or binaries with Segmentwright Scribe,
retain the applicable copyright notices, complete license texts, and notice
files from those exact distributions. This inventory does not replace them.
