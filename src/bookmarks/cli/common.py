import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any


def configure_logging() -> None:
    """Configure logging based on LOG_LEVEL."""
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.basicConfig(level=log_level, format="%(levelname)s %(message)s")


def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    """Convert argparse.Namespace into a JSON-serializable dict."""
    serialized: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in {"command", "command_handler"}:
            continue
        if isinstance(value, Path):
            serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized


def dump_cli_args(args: argparse.Namespace) -> int:
    payload = serialize_args(args)
    serialized = json.dumps(payload, indent=2 if args.pretty else None)
    print(serialized)
    return 0


def build_help_handler(parser: argparse.ArgumentParser):
    def _handler(_: argparse.Namespace) -> int:
        parser.print_help()
        return 0

    return _handler
