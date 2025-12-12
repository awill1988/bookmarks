import argparse
from pathlib import Path

from bookmarks.cli import common, schema
from bookmarks.cli.schema import DEFAULT_SCHEMA_MODEL, SCHEMA_MODEL_URL_ENV


def register_model_commands(subparsers: argparse._SubParsersAction) -> None:
    models_parser = subparsers.add_parser(
        "models",
        help="manage local model assets",
    )
    models_parser.set_defaults(command_handler=common.build_help_handler(models_parser))
    models_subparsers = models_parser.add_subparsers(dest="models_command")

    fetch_parser = models_subparsers.add_parser(
        "fetch",
        help="download a GGUF model to the local models directory",
    )
    fetch_parser.add_argument(
        "--url",
        help=f"model download url (or set {SCHEMA_MODEL_URL_ENV})",
    )
    fetch_parser.add_argument(
        "--path",
        type=Path,
        help=f"where to place the downloaded model (default: {DEFAULT_SCHEMA_MODEL})",
    )
    fetch_parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing model file",
    )
    fetch_parser.set_defaults(command_handler=schema.run_fetch_model)
