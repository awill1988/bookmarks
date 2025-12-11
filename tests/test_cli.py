import argparse
import json
import sqlite3
import sys
import types
from pathlib import Path

from bookmarks import cli


def test_run_export_torch_writes_payload(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "vectors.db"
    output_path = tmp_path / "vectors.pt"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            create table bookmark_embeddings (
                id integer primary key,
                url text not null,
                title text,
                vector json not null
            )
            """
        )
        conn.execute(
            """
            insert into bookmark_embeddings (url, title, vector)
            values (?, ?, ?)
            """,
            ("https://example.com", "Example", json.dumps([0.1, 0.2])),
        )

    saved: dict[str, object] = {}
    torch_module = types.SimpleNamespace()
    torch_module.float32 = "float32"
    torch_module.tensor = lambda data, dtype=None: data

    def fake_save(payload: dict[str, object], path: Path) -> None:
        saved["payload"] = payload
        saved["path"] = path

    torch_module.save = fake_save
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    args = argparse.Namespace(db_path=db_path, output=output_path)
    assert cli.run_export_torch(args) == 0
    assert saved["path"] == output_path
    assert saved["payload"] == {
        "embeddings": [[0.1, 0.2]],
        "bookmarks": [{"url": "https://example.com", "title": "Example"}],
    }
