import argparse
import logging
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from langchain_community.llms.llamacpp import LlamaCpp

from bookmarks.graphs.schema import DEFAULT_SCHEMA_SQL_PATH, build_schema_graph

SCHEMA_PROMPT = (
    "You are designing a deterministic sqlite schema for bookmarking. "
    "Include url, title, and a vector column for embeddings. "
    "Return a single CREATE TABLE statement with column types and constraints. "
    "Keep the response concise and reproducible."
)

SCHEMA_MODEL_ENV = "BOOKMARKS_MODEL_PATH"
SCHEMA_MODEL_URL_ENV = "BOOKMARKS_MODEL_URL"
DEFAULT_SCHEMA_MODEL = Path("models/schema.llama3.gguf")


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
