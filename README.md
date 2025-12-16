# bookmarks

Agent-driven CLI for organizing bookmarks using temporal embeddings.

Combines semantic text embeddings with normalized temporal features to cluster bookmarks by both content similarity and time period. See [DESIGN.md](DESIGN.md) for implementation details and research foundations.

## Setup

**Prerequisites**: Nix and direnv installed.

```bash
direnv allow              # or: nix develop
uv sync                   # base dependencies
uv sync --group torch     # add torch for visualization commands
```

## Command reference

### Generation commands

| Command | Description | Example |
|---------|-------------|---------|
| `gen export` | Generate embeddings from bookmark JSON and store in SQLite | `uv run bookmarks gen export -i bookmarks.json --db-path vectors.db` |
| `gen torch` | Export SQLite embeddings to Torch artifact | `uv run bookmarks gen torch --db-path vectors.db --output vectors.pt` |
| `gen schema-graph` | Infer JSON schema and generate SQL DDL via LangGraph | `uv run bookmarks gen schema-graph -i bookmarks.json --output schema.sql` |

### Visualization commands

Requires `--group torch` dependencies.

| Command | Description | Example |
|---------|-------------|---------|
| `vis summary` | Inspect torch artifact metadata | `uv run bookmarks vis summary --artifact vectors.pt --limit 5` |
| `vis organize` | Multi-resolution clustering for organization (by year, quarter, month) | `uv run bookmarks vis organize --artifact vectors.pt --resolutions all,year,quarter --output organize.json` |
| `vis neighbors` | Find nearest neighbors by cosine similarity with dates | `uv run bookmarks vis neighbors --artifact vectors.pt --index 0 --top-k 5` |

### Common flags

- `--artifact PATH`: Torch file to analyze (default: `vectors.pt`)
- `--db-path PATH`: SQLite database path (default: `vectors.db`)
- `--clusters N`: Number of k-means clusters (default: 6)
- `--include-stop-words`: Include stop words in token analysis (default: filter them)

## Quick workflow

```bash
# 1. generate embeddings with temporal features
uv run bookmarks gen export -i bookmarks.json --db-path vectors.db

# 2. export to torch for visualization
uv run bookmarks gen torch --db-path vectors.db --output vectors.pt

# 3. organize bookmarks (multi-resolution clustering)
uv run bookmarks vis organize --artifact vectors.pt --resolutions all,year --output organize.json

# 4. find similar bookmarks
uv run bookmarks vis neighbors --artifact vectors.pt --index 0
```
