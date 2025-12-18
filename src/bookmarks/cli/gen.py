import argparse
from pathlib import Path

from bookmarks.cli import common
from bookmarks.graphs.embed import DEFAULT_DB_PATH

DEFAULT_SCHEMA_SQL_PATH = Path("data/schema.sql")


def _run_demo_graph(args: argparse.Namespace) -> int:
    from bookmarks.cli import embed

    return embed.run_demo_graph(args)


def _run_export_torch(args: argparse.Namespace) -> int:
    from bookmarks.cli import embed

    return embed.run_export_torch(args)


def _run_schema_stub(args: argparse.Namespace) -> int:
    from bookmarks.cli import schema

    return schema.run_schema_stub(args)


def _run_schema_graph(args: argparse.Namespace) -> int:
    from bookmarks.cli import schema

    return schema.run_schema_graph(args)


def register_gen_commands(subparsers: argparse._SubParsersAction) -> None:
    gen_parser = subparsers.add_parser(
        "gen",
        help="generate bookmark artifacts via workflows",
    )
    gen_parser.set_defaults(command_handler=common.build_help_handler(gen_parser))
    gen_subparsers = gen_parser.add_subparsers(dest="gen_command")

    export_parser = gen_subparsers.add_parser(
        "export",
        help="embed bookmarks and persist vectors to sqlite (demo)",
    )
    export_parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="path to the raw bookmark export to ingest",
    )
    export_parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"where to write sqlite embeddings (default: data/vectors.db)",
    )
    export_parser.set_defaults(command_handler=_run_demo_graph)

    schema_parser = gen_subparsers.add_parser(
        "schema",
        help="generate a bookmark schema via a local llama-cpp model",
    )
    schema_parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="path to the raw bookmark export to inform schema generation",
    )
    schema_parser.add_argument(
        "--output",
        type=Path,
        help="optional path to write the schema recommendation",
    )
    schema_parser.set_defaults(command_handler=_run_schema_stub)

    schema_graph_parser = gen_subparsers.add_parser(
        "schema-graph",
        help="infer a JSON Schema and synthesize SQL via a workflow",
    )
    schema_graph_parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="path to the raw bookmark export to inspect",
    )
    schema_graph_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SCHEMA_SQL_PATH,
        help=f"where to write the generated sql (default: data/schema.sql)",
    )
    schema_graph_parser.set_defaults(command_handler=_run_schema_graph)

    torch_parser = gen_subparsers.add_parser(
        "torch",
        help="serialize sqlite embeddings into a torch artifact",
    )
    torch_parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"source sqlite embeddings (default: data/vectors.db)",
    )
    torch_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/vectors.pt"),
        help="where to write the torch artifact (default: data/vectors.pt)",
    )
    torch_parser.set_defaults(command_handler=_run_export_torch)
