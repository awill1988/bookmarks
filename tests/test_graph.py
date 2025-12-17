import json
import sqlite3
from pathlib import Path

import pytest

from bookmarks.graphs import embed as embed_graph
from bookmarks.graphs import schema as schema_graph


def test_normalize_payload_recurses_nested_collections() -> None:
    payload = {
        "folders": [
            {"items": [{"href": "https://example.com", "name": "Example"}]},
            {
                "child": {
                    "uri": "https://example.org",
                    "text": "Org",
                }
            },
        ]
    }

    normalized = embed_graph.normalize_payload(payload)

    assert normalized == [
        {"url": "https://example.com", "title": "Example"},
        {"url": "https://example.org", "title": "Org"},
    ]


def test_demo_graph_persists_embeddings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class DummyEmbedder:
        def __init__(self, model_name: str):  # noqa: D401
            self.model_name = model_name

        def embed_documents(self, docs: list[str]) -> list[list[float]]:
            return [[float(index)] for index, _ in enumerate(docs)]

    source = tmp_path / "bookmarks.json"
    db_path = tmp_path / "vectors.db"
    source.write_text(
        json.dumps(
            [
                {"url": "https://example.com", "title": "Example"},
                {"url": "https://example.org"},
            ]
        ),
        encoding="utf-8",
    )

    compiled = embed_graph.build_demo_graph(model_name="dummy", embedder_factory=DummyEmbedder)
    final_state = compiled.invoke({"source_path": source, "db_path": db_path})

    assert final_state["db_path"] == db_path
    with sqlite3.connect(db_path) as conn:
        rows = list(conn.execute("select url, title, vector from bookmark_embeddings order by id"))
    assert rows == [
        ("https://example.com", "Example", "[0.0]"),
        ("https://example.org", None, "[1.0]"),
    ]


def test_schema_graph_writes_sql(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class DummyLlama:
        def __init__(self, model_path: str, temperature: float, n_ctx: int, n_batch: int, n_threads: int, verbose: bool):  # noqa: D401
            self.model_path = model_path
            self.temperature = temperature
            self.n_ctx = n_ctx
            self.n_batch = n_batch
            self.n_threads = n_threads
            self.verbose = verbose

        def invoke(self, prompt: str) -> str:
            assert "JSON Schema" in prompt
            return "create table bookmarks (url text primary key);"

    source = tmp_path / "bookmarks.json"
    output = tmp_path / "schema.sql"
    source.write_text(
        json.dumps([{"url": "https://example.com", "title": "Example", "folder": "news"}]),
        encoding="utf-8",
    )

    compiled = schema_graph.build_schema_graph("dummy", output_path=output, llm_factory=DummyLlama)
    state = compiled.invoke({"source_path": source})

    assert state["output_path"] == output
    assert output.read_text(encoding="utf-8") == "create table bookmarks (url text primary key);"
