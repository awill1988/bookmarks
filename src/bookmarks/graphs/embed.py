import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Callable, Protocol, TypedDict

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langgraph.graph import END, StateGraph


DEFAULT_DB_PATH = Path("vectors.db")


class EmbeddingModel(Protocol):
    def embed_documents(self, docs: list[str]) -> list[list[float]]:
        ...


class GraphState(TypedDict, total=False):
    source_path: Path
    db_path: Path
    bookmarks: list[dict[str, str]]
    embeddings: list[list[float]]


def normalize_payload(payload: Any) -> list[dict[str, str]]:
    """Coerce mixed bookmark exports into a predictable list of url/title pairs."""

    normalized: list[dict[str, str]] = []

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item)
            return

        if isinstance(node, dict):
            # Treat dicts with link-like keys as bookmark candidates.
            url = str(
                node.get("url")
                or node.get("href")
                or node.get("link")
                or node.get("uri")
                or ""
            ).strip()
            title = str(
                node.get("title")
                or node.get("name")
                or node.get("text")
                or node.get("label")
                or ""
            ).strip()

            if url:
                normalized_entry: dict[str, str] = {"url": url}
                if title:
                    normalized_entry["title"] = title
                normalized.append(normalized_entry)

            # Recurse into child collections for nested bookmark formats.
            for value in node.values():
                visit(value)
            return

        logging.debug("skipping non-iterable node in payload")

    visit(payload)

    if not normalized:
        logging.warning("bookmark payload did not yield any entries; check format")

    return normalized


def load_bookmarks_node(state: GraphState) -> GraphState:
    source_path = Path(state["source_path"])
    if not source_path.exists():
        raise FileNotFoundError(f"bookmark source {source_path} does not exist")

    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    normalized = normalize_payload(payload)
    return {"bookmarks": normalized}


def embed_bookmarks_node(model: EmbeddingModel):
    def _embed(state: GraphState) -> GraphState:
        bookmarks = state.get("bookmarks", [])
        if not bookmarks:
            logging.warning("no bookmarks to embed")
            return {"embeddings": []}

        corpus = [entry.get("title") or entry["url"] for entry in bookmarks]
        embeddings = model.embed_documents(corpus)
        return {"embeddings": embeddings}

    return _embed


def persist_sqlite_node(state: GraphState) -> GraphState:
    db_path = Path(state.get("db_path") or DEFAULT_DB_PATH)
    bookmarks = state.get("bookmarks") or []
    embeddings = state.get("embeddings") or []

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            create table if not exists bookmark_embeddings (
                id integer primary key,
                url text not null,
                title text,
                vector json not null
            )
            """
        )
        for entry, vector in zip(bookmarks, embeddings, strict=False):
            conn.execute(
                """
                insert into bookmark_embeddings (url, title, vector)
                values (?, ?, ?)
                """,
                (entry["url"], entry.get("title"), json.dumps(vector)),
            )

    return {"db_path": db_path}


def build_demo_graph(
    model_name: str = "BAAI/bge-small-en-v1.5",
    embedder_factory: Callable[[str], EmbeddingModel] = FastEmbedEmbeddings,
):
    """Create a LangGraph pipeline to embed bookmarks into sqlite."""
    graph = StateGraph(GraphState)
    embedder = embedder_factory(model_name=model_name)

    graph.add_node("load", load_bookmarks_node)
    graph.add_node("embed", embed_bookmarks_node(embedder))
    graph.add_node("persist", persist_sqlite_node)

    graph.set_entry_point("load")
    graph.add_edge("load", "embed")
    graph.add_edge("embed", "persist")
    graph.add_edge("persist", END)

    return graph.compile()
