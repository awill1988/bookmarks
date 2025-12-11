# bookmarks

Centralize and evolve your bookmarks.

## Quickstart (nix + direnv)

- Prereqs: Nix and direnv installed.
- Allow the flake env to load: `direnv allow` (or run `nix develop` manually).
- Install deps inside the shell: `uv sync` (add `--group torch` to enable Torch-based commands).
- Run the demo export flow: `uv run bookmarks gen export -i bookmarks.json --db-path vectors.db`.
- Convert stored embeddings into a Torch artifact (requires Torch group): `uv run bookmarks gen torch --db-path vectors.db --output vectors.pt`.
- Explore the `.pt` artifact with Jupyter via Docker: `docker compose up torch_playground` then open `http://localhost:8888` and load `/workspace/vectors.pt`.
- Infer a JSON Schema and synthesize a bookmark SQL schema via LangGraph: `uv run bookmarks gen schema-graph -i bookmarks.json --output schema.sql`.
