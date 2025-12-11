# bookmarks

Centralize and evolve your bookmarks.

## Quickstart (nix + direnv)

- Prereqs: Nix and direnv installed.
- Allow the flake env to load: `direnv allow` (or run `nix develop` manually).
- Install deps inside the shell: `uv sync`.
- Run the demo export flow: `uv run bookmarks gen export -i bookmarks.json --db-path demo.db`.
