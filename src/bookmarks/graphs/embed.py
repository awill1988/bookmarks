import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Callable, Protocol, TypedDict
from urllib.parse import urlparse, parse_qs

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langgraph.graph import END, StateGraph


DEFAULT_DB_PATH = Path("vectors.db")


def parse_url_components(url: str) -> dict[str, str | list[str]]:
    """Extract domain, path, and query parameters from URL."""
    try:
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)

        # Extract path segments (filter empty, remove file extensions)
        segments = [
            seg.rsplit(".", 1)[0] if "." in seg else seg
            for seg in parsed.path.split("/")
            if seg and seg not in {"index", "home", "default"}
        ]

        return {
            "scheme": parsed.scheme,
            "domain": parsed.netloc,
            "path": parsed.path,
            "path_segments": segments[:3],  # Limit to first 3 segments to reduce noise
            "query_params": list(query_params.keys()),  # Just param names, not values
        }
    except Exception:
        return {"scheme": "", "domain": "", "path": "", "path_segments": [], "query_params": []}


class EmbeddingModel(Protocol):
    def embed_documents(self, docs: list[str]) -> list[list[float]]:
        ...


class GraphState(TypedDict, total=False):
    source_path: Path
    db_path: Path
    bookmarks: list[dict[str, str | int | None]]
    embeddings: list[list[float]]


def normalize_payload(payload: Any) -> list[dict[str, str | int | None]]:
    """Coerce mixed bookmark exports into a predictable list of url/title/timestamp entries."""

    normalized: list[dict[str, str | int | None]] = []

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

            # Extract timestamp (dateAdded in microseconds for Firefox exports)
            date_added = node.get("dateAdded") or node.get("date_added") or node.get("timestamp") or node.get("created")
            timestamp = int(date_added) if date_added else None

            if url:
                # Parse URL components
                url_parts = parse_url_components(url)

                normalized_entry: dict[str, str | int | None] = {"url": url}
                if title:
                    normalized_entry["title"] = title
                if timestamp:
                    normalized_entry["timestamp"] = timestamp

                # Store parsed URL components
                normalized_entry["domain"] = url_parts["domain"]
                normalized_entry["path"] = url_parts["path"]
                normalized_entry["path_segments"] = json.dumps(url_parts["path_segments"])
                normalized_entry["query_params"] = json.dumps(url_parts["query_params"])

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


def _compute_temporal_features(timestamps: list[int | None]) -> list[list[float]]:
    """Convert timestamps to normalized temporal features for embedding concatenation."""
    # Filter valid timestamps
    valid_ts = [ts for ts in timestamps if ts is not None]

    if not valid_ts:
        # No valid timestamps - return neutral features (0.5) for all
        return [[0.5] for _ in timestamps]

    # Convert microseconds to seconds for Firefox bookmarks
    # Firefox uses microseconds since epoch
    ts_seconds = [ts / 1_000_000 if ts and ts > 10_000_000_000 else ts for ts in valid_ts]
    min_ts = min(ts_seconds)
    max_ts = max(ts_seconds)
    range_ts = max_ts - min_ts if max_ts > min_ts else 1.0

    # Normalize each timestamp to 0-1 range (or 0.5 if missing)
    features = []
    for ts in timestamps:
        if ts is None:
            features.append([0.5])  # Neutral value for missing timestamps
        else:
            ts_sec = ts / 1_000_000 if ts > 10_000_000_000 else ts
            normalized = (ts_sec - min_ts) / range_ts
            features.append([normalized])

    return features


def embed_bookmarks_node(model: EmbeddingModel):
    def _embed(state: GraphState) -> GraphState:
        bookmarks = state.get("bookmarks", [])
        if not bookmarks:
            logging.warning("no bookmarks to embed")
            return {"embeddings": []}

        # Include domain and path segments in embedding text for better clustering
        corpus = []
        for entry in bookmarks:
            title = entry.get("title") or ""
            domain = entry.get("domain") or ""
            path_segments_json = entry.get("path_segments") or "[]"
            url = entry.get("url") or ""

            try:
                path_segments = json.loads(path_segments_json)
            except (json.JSONDecodeError, TypeError):
                path_segments = []

            # Build text: title + domain + path segments
            parts = [title] if title else []
            if domain:
                parts.append(domain)
            if path_segments:
                parts.extend(path_segments)

            text = " ".join(parts) if parts else url
            corpus.append(text)

        text_embeddings = model.embed_documents(corpus)

        # Extract timestamps and compute temporal features
        timestamps = [entry.get("timestamp") for entry in bookmarks]
        temporal_features = _compute_temporal_features(timestamps)

        # Concatenate text embeddings with temporal features
        embeddings = [
            text_emb + temp_feat
            for text_emb, temp_feat in zip(text_embeddings, temporal_features, strict=True)
        ]

        logging.info("embedded %s bookmarks with temporal features (dim: %s)", len(embeddings), len(embeddings[0]) if embeddings else 0)
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
                timestamp integer,
                domain text,
                path text,
                path_segments json,
                query_params json,
                vector json not null
            )
            """
        )
        for entry, vector in zip(bookmarks, embeddings, strict=False):
            conn.execute(
                """
                insert into bookmark_embeddings (url, title, timestamp, domain, path, path_segments, query_params, vector)
                values (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry["url"],
                    entry.get("title"),
                    entry.get("timestamp"),
                    entry.get("domain"),
                    entry.get("path"),
                    entry.get("path_segments"),
                    entry.get("query_params"),
                    json.dumps(vector),
                ),
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
