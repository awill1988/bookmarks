import argparse
from pathlib import Path

from bookmarks.cli import common, gen, vis
from bookmarks.tracing import configure_tracing


def run_demo_graph(args: argparse.Namespace) -> int:
    from bookmarks.cli.embed import run_demo_graph as impl

    return impl(args)


def run_export_torch(args: argparse.Namespace) -> int:
    from bookmarks.cli.embed import run_export_torch as impl

    return impl(args)


def run_schema_stub(args: argparse.Namespace) -> int:
    from bookmarks.cli.schema import run_schema_stub as impl

    return impl(args)


def run_schema_graph(args: argparse.Namespace) -> int:
    from bookmarks.cli.schema import run_schema_graph as impl

    return impl(args)


def run_vis_torch(args: argparse.Namespace) -> int:
    from bookmarks.cli.vis import run_vis_torch as impl

    return impl(args)


def run_vis_cluster(args: argparse.Namespace) -> int:
    from bookmarks.cli.vis import run_vis_cluster as impl

    return impl(args)


def run_vis_organize(args: argparse.Namespace) -> int:
    from bookmarks.cli.vis import run_vis_organize as impl

    return impl(args)


def run_vis_neighbors(args: argparse.Namespace) -> int:
    from bookmarks.cli.vis import run_vis_neighbors as impl

    return impl(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "agent-driven bookmarks scaffold: inspect cli arguments or generate artifacts"
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    help_parser = subparsers.add_parser("help", help="show cli usage and exit")
    help_parser.set_defaults(command_handler=common.build_help_handler(parser))

    args_parser = subparsers.add_parser(
        "dump-args",
        help="echo parsed cli arguments as json for debugging",
    )
    args_parser.add_argument(
        "--source",
        type=Path,
        help="path to the raw bookmark export to ingest",
    )
    args_parser.add_argument(
        "--output",
        type=Path,
        help="destination for the enriched bookmark payload",
    )
    args_parser.add_argument(
        "--prompt",
        type=Path,
        help="prompt file guiding enrichment or grouping",
    )
    args_parser.add_argument(
        "--run-mode",
        choices=["enrich", "group"],
        default="enrich",
        help="select enrichment mode: fill missing metadata or group via prompt",
    )
    args_parser.add_argument(
        "--pretty",
        action="store_true",
        help="pretty-print the arg dump",
    )
    args_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse arguments without dispatching any workflow",
    )
    args_parser.set_defaults(command_handler=common.dump_cli_args)

    gen.register_gen_commands(subparsers)
    vis.register_vis_command(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    common.configure_logging()
    configure_tracing()

    command_handler = getattr(args, "command_handler", None)
    if command_handler is None:
        return common.build_help_handler(parser)(args)

    return command_handler(args)


__all__ = [
    "main",
    "build_parser",
    "run_demo_graph",
    "run_schema_stub",
    "run_schema_graph",
    "run_export_torch",
    "run_vis_torch",
    "run_vis_cluster",
    "run_vis_organize",
    "run_vis_neighbors",
]
