import json
import sqlite3
from pathlib import Path

import pytest

from bookmarks import graph


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

    normalized = graph.normalize_payload(payload)

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

    monkeypatch.setattr(graph, "FastEmbedEmbeddings", DummyEmbedder)

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

    compiled = graph.build_demo_graph(model_name="dummy")
    final_state = compiled.invoke({"source_path": source, "db_path": db_path})

    assert final_state["db_path"] == db_path
    with sqlite3.connect(db_path) as conn:
        rows = list(conn.execute("select url, title, vector from bookmark_embeddings order by id"))
    assert rows == [
        ("https://example.com", "Example", "[0.0]"),
        ("https://example.org", None, "[1.0]"),
    ]
