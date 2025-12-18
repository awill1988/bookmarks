import argparse
import logging
import os
from pathlib import Path

from bookmarks.agents.llama_cpp_model import LlamaCppChatModel
from bookmarks.agents.schema_sql_agent import generate_schema_sql_with_retry
from bookmarks.graphs.schema import DEFAULT_SCHEMA_SQL_PATH, build_schema_graph
from bookmarks.models.gguf import DEFAULT_CACHE_DIR, ensure_gguf_model

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
        model = LlamaCppChatModel(model_path, verbose=False)
    except Exception as exc:
        raise RuntimeError(f"failed to initialize local model: {exc}") from exc

    return generate_schema_sql_with_retry(model, prompt)


def ensure_schema_model() -> Path:
    """ensure schema model is available, downloading if necessary."""
    repo_id = os.getenv("BOOKMARKS_SCHEMA_REPO_ID")
    filename = os.getenv("BOOKMARKS_SCHEMA_FILENAME")

    try:
        return ensure_gguf_model(
            repo_id=repo_id,
            filename=filename,
            cache_dir=DEFAULT_CACHE_DIR,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise FileNotFoundError(f"failed to resolve schema model: {exc}") from exc




def run_schema_stub(args: argparse.Namespace) -> int:
    prompt = build_schema_prompt(args.input)
    try:
        model_path = ensure_schema_model()
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
        model_path = ensure_schema_model()
        graph = build_schema_graph(model_path, output_path=args.output)
        final_state = graph.invoke({"source_path": args.input, "output_path": args.output})
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        logging.error("schema graph failed: %s", exc)
        return 1

    destination = final_state.get("output_path") or args.output or DEFAULT_SCHEMA_SQL_PATH
    logging.info("schema graph wrote sql to %s", destination)
    return 0
