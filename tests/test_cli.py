import argparse
import json
import sqlite3
import sys
import types
from pathlib import Path
import logging

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


def test_run_vis_torch_reports_summary(monkeypatch, tmp_path: Path, caplog) -> None:
    artifact = tmp_path / "vectors.pt"
    artifact.touch()

    class DummyTensor:
        def __init__(self, data):
            self.data = data

        @property
        def shape(self):
            return (len(self.data), len(self.data[0]) if self.data else 0)

        @property
        def ndim(self):
            return 2

    payload = {
        "embeddings": DummyTensor([[0.1, 0.2], [0.3, 0.4]]),
        "bookmarks": [
            {"url": "https://example.com", "title": "Example"},
            {"url": "https://example.org", "title": "Org"},
        ],
    }

    torch_module = types.SimpleNamespace()
    torch_module.Tensor = DummyTensor
    torch_module.tensor = lambda data: DummyTensor(data)
    torch_module.load = lambda path, map_location=None: payload

    monkeypatch.setitem(sys.modules, "torch", torch_module)

    caplog.set_level(logging.INFO)
    args = argparse.Namespace(artifact=artifact, limit=1, vis_command="summary")
    result = cli.run_vis_torch(args)
    assert result == 0
    assert any("loaded 2 embeddings" in message for message in caplog.text.splitlines())


def test_run_vis_cluster(monkeypatch, tmp_path: Path, caplog) -> None:
    artifact = tmp_path / "vectors.pt"
    artifact.touch()

    class DummyTensor:
        def __init__(self, data):
            self.data = data

        @property
        def shape(self):
            return (len(self.data), len(self.data[0]) if self.data else 0)

        @property
        def ndim(self):
            return 2

        def clone(self):
            return DummyTensor([row[:] for row in self.data])

        def mean(self, dim=0):
            cols = list(zip(*self.data))
            return DummyTensor([[sum(col) / len(col) for col in cols]])

        def __sub__(self, other):
            other_row = other.data[0]
            return DummyTensor([[a - b for a, b in zip(row, other_row)] for row in self.data])

        def __matmul__(self, other):
            result = []
            other_t = list(zip(*other.data))
            for row in self.data:
                result.append([sum(a * b for a, b in zip(row, col)) for col in other_t])
            return DummyTensor(result)

        def any(self):
            return any(self.data)

        def argmin(self, dim=0):
            # simple argmin across rows
            mins = []
            for row in self.data:
                mins.append(min(range(len(row)), key=lambda idx: row[idx]))
            return DummyTensor([[m] for m in mins])

        def __iter__(self):
            return iter(self.data)

    payload = {
        "embeddings": DummyTensor([[0.1, 0.2], [0.3, 0.4]]),
        "bookmarks": [{"url": "a", "title": "A title"}, {"url": "b", "title": "B title"}],
    }

    torch_module = types.SimpleNamespace()
    torch_module.Tensor = DummyTensor
    torch_module.tensor = lambda data: DummyTensor(data)
    torch_module.load = lambda path, map_location=None: payload
    torch_module.cdist = lambda a, b, p=2: DummyTensor([[0, 1], [1, 0]])

    caplog.set_level(logging.INFO)
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    args = argparse.Namespace(artifact=artifact, limit=2, clusters=2, top_tokens=2, vis_command="cluster")
    result = cli.run_vis_cluster(args)
    assert result == 0
    assert any("cluster" in line for line in caplog.text.splitlines())


def test_run_vis_neighbors(monkeypatch, tmp_path: Path, caplog) -> None:
    artifact = tmp_path / "vectors.pt"
    artifact.touch()

    class DummyTensor:
        def __init__(self, data):
            self.data = data

        @property
        def shape(self):
            return (len(self.data), len(self.data[0]) if self.data else 0)

        def __getitem__(self, item):
            return self.data[item]

        def __iter__(self):
            return iter(self.data)

    payload = {
        "embeddings": DummyTensor([[1.0, 0.0], [0.5, 0.5]]),
        "bookmarks": [{"url": "a", "title": "A"}, {"url": "b", "title": "B"}],
    }

    torch_module = types.SimpleNamespace()
    torch_module.Tensor = DummyTensor
    torch_module.tensor = lambda data: DummyTensor(data)
    torch_module.load = lambda path, map_location=None: payload

    def fake_normalize(tensor, dim=1):
        return tensor

    def fake_matmul(a, b):
        # simple dot products vs a vector
        return [sum(x * y for x, y in zip(row, b)) for row in a]

    torch_module.nn = types.SimpleNamespace(functional=types.SimpleNamespace(normalize=fake_normalize))
    torch_module.matmul = fake_matmul

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    caplog.set_level(logging.INFO)

    args = argparse.Namespace(artifact=artifact, index=0, top_k=1, vis_command="neighbors")
    result = cli.run_vis_neighbors(args)
    assert result == 0
    assert any("neighbors for" in line for line in caplog.text.splitlines())
