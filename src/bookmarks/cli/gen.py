import argparse
from pathlib import Path

from bookmarks.cli import common, embed, schema
from bookmarks.graphs.embed import DEFAULT_DB_PATH
from bookmarks.graphs.schema import DEFAULT_SCHEMA_SQL_PATH


def register_gen_commands(subparsers: argparse._SubParsersAction) -> None:
    gen_parser = subparsers.add_parser(
        "gen",
        help="generate bookmark artifacts via langgraph workflows",
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
        help=f"where to write sqlite embeddings (default: {DEFAULT_DB_PATH})",
    )
    export_parser.set_defaults(command_handler=embed.run_demo_graph)

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
    schema_parser.set_defaults(command_handler=schema.run_schema_stub)

    schema_graph_parser = gen_subparsers.add_parser(
        "schema-graph",
        help="infer a JSON Schema and synthesize SQL via a LangGraph workflow",
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
        help=f"where to write the generated sql (default: {DEFAULT_SCHEMA_SQL_PATH})",
    )
    schema_graph_parser.set_defaults(command_handler=schema.run_schema_graph)

    torch_parser = gen_subparsers.add_parser(
        "torch",
        help="serialize sqlite embeddings into a torch artifact",
    )
    torch_parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"source sqlite embeddings (default: {DEFAULT_DB_PATH})",
    )
    torch_parser.add_argument(
        "--output",
        type=Path,
        default=Path("vectors.pt"),
        help="where to write the torch artifact (default: vectors.pt)",
    )
    torch_parser.set_defaults(command_handler=embed.run_export_torch)
