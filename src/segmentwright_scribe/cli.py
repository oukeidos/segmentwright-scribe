from __future__ import annotations

import sys
import traceback

from .config import ConfigError, build_config, build_parser
from .pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    verbose = bool(getattr(args, "verbose", False))
    try:
        config = build_config(args)
        run_pipeline(config)
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI must report failures cleanly.
        if verbose:
            traceback.print_exception(exc, file=sys.stderr)
        else:
            prefix = "segmentwright-scribe"
            if isinstance(exc, ConfigError):
                print(f"{prefix}: {exc}", file=sys.stderr)
            else:
                print(f"{prefix}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
