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

### GPU Acceleration

GPU acceleration is **automatically configured** when entering the nix shell:
- **macOS (Apple Silicon)**: Metal backend
- **Linux with NVIDIA GPU**: CUDA backend
- **Linux/Other**: Vulkan fallback

The flake detects your hardware and sets `CMAKE_ARGS` automatically. After entering the shell, rebuild llama-cpp-python once:

```bash
uv pip install --force-reinstall --no-cache-dir llama-cpp-python
```

This provides **5-10x speedup** for schema generation.

**Disable GPU**: `export BOOKMARKS_FORCE_CPU=1` before running commands.

## Model Configuration

Schema generation commands automatically download GGUF models from HuggingFace Hub on first use. Models are cached in the `models/` directory.

**Default model**: TheBloke/Llama-2-7B-Chat-GGUF (Q4_K_M quantization, ~4GB)

**Override defaults** via environment variables:
```bash
export BOOKMARKS_SCHEMA_REPO_ID="TheBloke/CodeLlama-7B-GGUF"
export BOOKMARKS_SCHEMA_FILENAME="codellama-7b.Q4_K_M.gguf"
```

**First-time usage**: Models download automatically when running:
```bash
uv run bookmarks gen schema -i bookmarks.json
uv run bookmarks gen schema-graph -i bookmarks.json --output schema.sql
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
