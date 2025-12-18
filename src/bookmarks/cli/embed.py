import argparse
import json
import logging
import sqlite3
from pathlib import Path

from bookmarks.graphs.embed import GraphState, build_demo_graph


def load_sqlite_embeddings(db_path: Path) -> tuple[list[dict[str, str | int | None]], list[list[float]]]:
    if not db_path.exists():
        raise FileNotFoundError(f"expected sqlite store at {db_path}")

    bookmarks: list[dict[str, str | int | None]] = []
    vectors: list[list[float]] = []

    with sqlite3.connect(db_path) as conn:
        try:
            rows = conn.execute(
                "select url, title, timestamp, vector from bookmark_embeddings order by id"
            )
            for url, title, timestamp, vector_text in rows:
                try:
                    vector = json.loads(vector_text)
                except json.JSONDecodeError:
                    logging.warning("skipping malformed vector for %s", url)
                    continue
                entry: dict[str, str | int | None] = {"url": url, "title": title}
                if timestamp is not None:
                    entry["timestamp"] = timestamp
                bookmarks.append(entry)
                vectors.append(vector)
        except sqlite3.OperationalError:
            rows = conn.execute(
                "select url, title, vector from bookmark_embeddings order by id"
            )
            for url, title, vector_text in rows:
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
