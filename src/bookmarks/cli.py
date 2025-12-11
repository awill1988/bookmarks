import argparse
import json
import logging
import os
import shutil
import sqlite3
import urllib.error
import urllib.request
from functools import partial
from pathlib import Path
from typing import Any
from langchain_community.llms.llamacpp import LlamaCpp

from bookmarks.graphs.embed import DEFAULT_DB_PATH, GraphState, build_demo_graph
from bookmarks.graphs.schema import DEFAULT_SCHEMA_SQL_PATH, build_schema_graph


def configure_logging() -> None:
    """Configure logging based on LOG_LEVEL."""
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.basicConfig(level=log_level, format="%(levelname)s %(message)s")


def show_help(_: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    parser.print_help()
    return 0


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


def load_sqlite_embeddings(db_path: Path) -> tuple[list[dict[str, str | None]], list[list[float]]]:
    if not db_path.exists():
        raise FileNotFoundError(f"expected sqlite store at {db_path}")

    bookmarks: list[dict[str, str | None]] = []
    vectors: list[list[float]] = []

    with sqlite3.connect(db_path) as conn:
        for url, title, vector_text in conn.execute(
            "select url, title, vector from bookmark_embeddings order by id"
        ):
            try:
                vector = json.loads(vector_text)
            except json.JSONDecodeError:
                logging.warning("skipping malformed vector for %s", url)
                continue
            bookmarks.append({"url": url, "title": title})
            vectors.append(vector)

    return bookmarks, vectors


def run_demo_graph(args: argparse.Namespace) -> int:
    graph = build_demo_graph()

    initial_state: GraphState = {
        "source_path": args.input,
        "db_path": args.db_path,
    }
    final_state = graph.invoke(initial_state)

    stored = len(final_state.get("bookmarks") or [])
    logging.info("graph run complete; stored %s bookmarks to %s", stored, args.db_path)
    return 0


def run_export_torch(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        logging.error("torch not installed; install torch to run this command")
        return 1

    db_path = args.db_path
    output_path = args.output

    try:
        bookmarks, vectors = load_sqlite_embeddings(db_path)
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        return 1

    if not vectors:
        logging.error("no embeddings found in %s", db_path)
        return 1

    try:
        tensor = torch.tensor(vectors, dtype=torch.float32)
    except Exception as exc:
        logging.error("failed to convert embeddings to torch tensor: %s", exc)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save({"embeddings": tensor, "bookmarks": bookmarks}, output_path)
    except OSError as exc:
        logging.error("failed to write torch artifact: %s", exc)
        return 1

    logging.info("torch export wrote %s entries to %s", len(vectors), output_path)
    return 0


SCHEMA_PROMPT = (
    "You are designing a deterministic sqlite schema for bookmarking. "
    "Include url, title, and a vector column for embeddings. "
    "Return a single CREATE TABLE statement with column types and constraints. "
    "Keep the response concise and reproducible."
)


def build_schema_prompt(input_path: Path) -> str:
    sample = ""
    try:
        sample = input_path.read_text(encoding="utf-8")[:2000]
    except OSError:
        logging.warning("unable to read input for schema context; continuing without sample")

    context = f"\n\nsample export (truncated):\n{sample}" if sample else ""
    return f"{SCHEMA_PROMPT}{context}"


def generate_schema_local(prompt: str, model_path: Path) -> str:
    try:
        llm = LlamaCpp(
            model_path=str(model_path),
            temperature=0.0,
            n_threads=max(os.cpu_count() or 1, 1),
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(f"failed to initialize local model: {exc}") from exc

    try:
        result = llm.invoke(prompt)
    except Exception as exc:
        raise RuntimeError(f"model invocation failed: {exc}") from exc

    return str(result).strip()


SCHEMA_MODEL_ENV = "BOOKMARKS_MODEL_PATH"
SCHEMA_MODEL_URL_ENV = "BOOKMARKS_MODEL_URL"
DEFAULT_SCHEMA_MODEL = Path("models/schema.llama3.gguf")


def resolve_schema_model_path() -> Path:
    candidate = Path(os.getenv(SCHEMA_MODEL_ENV, "") or DEFAULT_SCHEMA_MODEL)
    if not candidate.exists():
        raise FileNotFoundError(f"expected schema model at {candidate}")
    return candidate


def resolve_schema_model_url(explicit_url: str | None) -> str:
    url = explicit_url or os.getenv(SCHEMA_MODEL_URL_ENV, "")
    if not url:
        raise ValueError("no model url provided; set BOOKMARKS_MODEL_URL or pass --url")
    return url


def fetch_model(url: str, destination: Path, overwrite: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination

    request = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"download failed: {exc}") from exc

    return destination


def run_fetch_model(args: argparse.Namespace) -> int:
    try:
        url = resolve_schema_model_url(args.url)
        destination = Path(args.path or DEFAULT_SCHEMA_MODEL)
        fetch_model(url, destination, overwrite=args.force)
    except (ValueError, RuntimeError) as exc:
        logging.error("model fetch failed: %s", exc)
        return 1

    logging.info("model fetched to %s", destination)
    return 0


def run_schema_stub(args: argparse.Namespace) -> int:
    prompt = build_schema_prompt(args.input)
    try:
        model_path = resolve_schema_model_path()
        schema_text = generate_schema_local(prompt, model_path)
    except (FileNotFoundError, ImportError, RuntimeError) as exc:
        logging.error("schema generation failed: %s", exc)
        return 1

    if not schema_text:
        logging.error("model returned empty schema recommendation")
        return 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(schema_text, encoding="utf-8")
        logging.info("schema recommendation written to %s", args.output)
    else:
        print(schema_text)

    return 0


def run_schema_graph(args: argparse.Namespace) -> int:
    try:
        model_path = resolve_schema_model_path()
        graph = build_schema_graph(model_path, output_path=args.output)
        final_state = graph.invoke({"source_path": args.input, "output_path": args.output})
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        logging.error("schema graph failed: %s", exc)
        return 1

    destination = final_state.get("output_path") or args.output or DEFAULT_SCHEMA_SQL_PATH
    logging.info("schema graph wrote sql to %s", destination)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "agent-driven bookmarks scaffold: inspect cli arguments or generate artifacts via langgraph"
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    help_parser = subparsers.add_parser("help", help="show cli usage and exit")
    help_parser.set_defaults(command_handler=partial(show_help, parser=parser))

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
    args_parser.set_defaults(command_handler=dump_cli_args)

    gen_parser = subparsers.add_parser(
        "gen",
        help="generate bookmark artifacts via langgraph workflows",
    )
    gen_parser.set_defaults(command_handler=partial(show_help, parser=gen_parser))
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
    export_parser.set_defaults(command_handler=run_demo_graph)

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
    torch_parser.set_defaults(command_handler=run_export_torch)

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
    schema_parser.set_defaults(command_handler=run_schema_stub)

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
    schema_graph_parser.set_defaults(command_handler=run_schema_graph)

    models_parser = subparsers.add_parser(
        "models",
        help="manage local model assets",
    )
    models_parser.set_defaults(command_handler=partial(show_help, parser=models_parser))
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
    fetch_parser.set_defaults(command_handler=run_fetch_model)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging()

    command_handler = getattr(args, "command_handler", None)
    if command_handler is None:
        return show_help(args, parser)

    return command_handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
