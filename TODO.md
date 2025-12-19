# Rust Refactor Plan: Schema Generation Only

Complete rewrite of bookmarks project focusing solely on schema generation. Delete all Python code and create a Rust workspace with OTEL observability for Langfuse.

## Overview

Port only the schema generation pipeline:
- Load bookmark JSON → Infer JSON schema → Derive field hints → LLM SQL generation → Extract DDL → Persist

Keep the same workflow logic, port to idiomatic Rust with custom agent loop (retry + error feedback).

## Workspace Structure

```
bookmarks/
├── Cargo.toml                    # workspace root
├── common/
│   ├── Cargo.toml               # shared library
│   └── src/
│       ├── lib.rs               # public exports
│       ├── schema/
│       │   ├── mod.rs           # schema module
│       │   ├── inference.rs     # JSON schema inference
│       │   └── hints.rs         # field hint derivation
│       ├── llm/
│       │   ├── mod.rs           # LLM module
│       │   ├── model.rs         # llama.cpp wrapper
│       │   ├── downloader.rs    # GGUF model download (HF Hub)
│       │   └── gpu.rs           # GPU detection/config
│       ├── agent/
│       │   ├── mod.rs           # agent module
│       │   ├── prompt.rs        # prompt templates
│       │   ├── executor.rs      # LLM execution with retry
│       │   └── parser.rs        # DDL extraction regex
│       ├── tracing/
│       │   ├── mod.rs           # tracing module
│       │   └── init.rs          # OTEL setup for Langfuse
│       └── error.rs             # error types
└── bookmarks/
    ├── Cargo.toml               # binary crate
    └── src/
        ├── main.rs              # CLI entry point
        └── cmd.rs               # clap command definitions
```

## Dependencies

### Workspace Root (Cargo.toml)

```toml
[workspace]
resolver = "2"
members = ["common", "bookmarks"]

[workspace.dependencies]
# CLI
clap = { version = "4", features = ["derive", "env"] }

# Async runtime
tokio = { version = "1.35", features = ["full"] }
tokio-util = "0.7"
futures = "0.3"

# Error handling
thiserror = "1.0"
anyhow = "1.0"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Observability
opentelemetry = { version = "0.28", features = ["trace"] }
opentelemetry_sdk = { version = "0.28", features = ["trace", "rt-tokio"] }
opentelemetry-resource-detectors = { version = "0.7" }
opentelemetry-otlp = { version = "0.28", default-features = false, features = ["trace", "grpc-tonic"] }
opentelemetry-semantic-conventions = { version = "0.28" }
tracing = "0.1"
tracing-core = "0.1"
tracing-subscriber = { version = "0.3", features = ["registry", "env-filter", "json"] }
tracing-opentelemetry = "0.29"

# LLM inference
llm = "0.2"  # llama.cpp Rust bindings (simpler than llama-cpp-rs)

# Model downloading
hf-hub = "0.3"  # HuggingFace Hub API

# Regex for DDL extraction
regex = "1.10"

# Memory allocator
mimalloc = "0.1"

# Path/filesystem
once_cell = "1.19"
```

**Dependency Rationale**:
- `llm` crate: Pure Rust llama.cpp bindings, actively maintained, supports GGUF
- `hf-hub`: Official Rust client for HuggingFace Hub (model downloading)
- OTEL versions 0.28/0.29 for ecosystem compatibility
- Skip candle - not needed for schema generation (no embedding, only LLM inference)

### Common Library (common/Cargo.toml)

```toml
[package]
name = "common"
version = "0.1.0"
edition = "2021"

[dependencies]
thiserror = { workspace = true }
anyhow = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
tokio = { workspace = true }
tokio-util = { workspace = true }
futures = { workspace = true }
regex = { workspace = true }
once_cell = { workspace = true }

# LLM
llm = { workspace = true }
hf-hub = { workspace = true }

# OTEL
opentelemetry = { workspace = true }
opentelemetry_sdk = { workspace = true }
opentelemetry-resource-detectors = { workspace = true }
opentelemetry-otlp = { workspace = true }
opentelemetry-semantic-conventions = { workspace = true }
tracing = { workspace = true }
tracing-core = { workspace = true }
tracing-subscriber = { workspace = true }
tracing-opentelemetry = { workspace = true }
```

### Binary Crate (bookmarks/Cargo.toml)

```toml
[package]
name = "bookmarks"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "bookmarks"
path = "src/main.rs"

[dependencies]
common = { path = "../common" }
clap = { workspace = true }
tokio = { workspace = true }
anyhow = { workspace = true }
mimalloc = { workspace = true }
tracing = { workspace = true }
```

## Key Implementation Details

### 1. JSON Schema Inference (common/src/schema/inference.rs)

Port Python logic from schema.py:85-125:
- `infer_type(value: &serde_json::Value) -> &str`
- `merge_types(lhs: Vec<String>, rhs: Vec<String>) -> Vec<String>`
- `merge_schema(lhs: Schema, rhs: Schema) -> Schema`
- `infer_json_schema(payload: &Value) -> Schema`

Return custom `Schema` struct (maps to JSON Schema format).

### 2. Field Hints (common/src/schema/hints.rs)

Port Python logic from schema.py:128-148:
- Walk schema properties
- Detect timestamp fields: "date", "time", "updated", "created"
- Detect folder fields: "folder", "path", "group", "category", "tag"
- Return `FieldHints { timestamps: Vec<String>, folders: Vec<String> }`

### 3. GPU Detection (common/src/llm/gpu.rs)

Port missing Python logic (from tests):
- `detect_gpu_backend() -> GpuBackend` enum (Metal, Cuda, Vulkan, Cpu)
- `get_gpu_config() -> GpuConfig { n_gpu_layers: i32, backend: GpuBackend, available: bool }`
- Check env: `BOOKMARKS_FORCE_CPU`, `BOOKMARKS_GPU_LAYERS`
- Platform detection: macOS arm64 → Metal, Linux NVIDIA → CUDA, fallback → Vulkan/CPU

### 4. Model Downloader (common/src/llm/downloader.rs)

Port missing Python logic (from tests):
- `ensure_gguf_model(repo_id: Option<String>, filename: Option<String>, cache_dir: Option<PathBuf>) -> Result<PathBuf>`
- Use `hf-hub` crate to download from HuggingFace
- Check env: `BOOKMARKS_SCHEMA_REPO_ID`, `BOOKMARKS_SCHEMA_FILENAME`
- Defaults: "TheBloke/Llama-2-7B-Chat-GGUF", "llama-2-7b-chat.Q4_K_M.gguf"
- Cache in `models/` directory

### 5. LLM Wrapper (common/src/llm/model.rs)

Port llama_cpp_model.py functionality:
- `LlamaModel::new(model_path: PathBuf, config: ModelConfig) -> Result<Self>`
- `ModelConfig { n_ctx: usize, n_batch: usize, temperature: f32, verbose: bool }`
- Load model with `llm` crate
- `generate(&self, messages: Vec<Message>) -> Result<String>`
- Apply GPU config from gpu.rs
- Instrument with OTEL spans (capture input/output)

### 6. Agent Executor (common/src/agent/executor.rs)

Port schema_sql_agent.py:80-95 retry logic:
- `generate_schema_sql_with_retry(model: &LlamaModel, prompt: String, max_attempts: usize) -> Result<String>`
- Retry loop: on failure, append error to prompt: "previous output was invalid: {err}"
- Trace each attempt with OTEL spans

### 7. DDL Parser (common/src/agent/parser.rs)

Port schema_sql_agent.py:18-33:
- `extract_create_table(sql_text: &str) -> Result<String>`
- Strip markdown fences: ` ```sql ... ``` `
- Find "CREATE TABLE" (case-insensitive regex)
- Validate existence, extract statement
- Return clean DDL

### 8. Prompt Templates (common/src/agent/prompt.rs)

Port schema_sql_agent.py:11-60:
- `SCHEMA_SYSTEM_PROMPT` constant
- `build_schema_task_prompt(schema: &Schema, hints: &FieldHints) -> String`
- Format JSON schema + hints into prompt text

### 9. OTEL Tracing (common/src/tracing/init.rs)

Port tracing.py:9-50:
- `init_tracing(service_name: &str) -> Result<OtelGuard>`
- Check env: `BOOKMARKS_ENABLE_TRACING`, `PHOENIX_COLLECTOR_ENDPOINT`
- Configure OTLP gRPC exporter to Langfuse endpoint
- Resource detection: OS, process, SDK
- Tracer provider with batch exporter
- Subscriber: JSON logging + OTEL layer
- Filter level from `LOG_LEVEL` env var
- Return guard for graceful shutdown

### 10. CLI (bookmarks/src/cmd.rs + main.rs)

Port cli/gen.py schema command:
- clap derive with `#[derive(Parser)]`
- Command: `bookmarks gen schema -i <input.json> -o <output.sql>`
- Args: `#[arg(env)]` for environment variable support
- Main flow:
  1. Init tracing
  2. Load JSON file
  3. Infer schema
  4. Derive hints
  5. Download/load model
  6. Generate SQL with retry
  7. Extract DDL
  8. Write to output file

### 11. Error Handling (common/src/error.rs)

Use thiserror:
```rust
#[derive(Error, Debug)]
pub enum BookmarksError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("model error: {0}")]
    Model(String),

    #[error("schema inference error: {0}")]
    Schema(String),

    #[error("ddl extraction failed: {0}")]
    DdlExtraction(String),
}
```

## Environment Variables

Support all existing env vars:
- `BOOKMARKS_SCHEMA_REPO_ID` - HF repo for model
- `BOOKMARKS_SCHEMA_FILENAME` - GGUF filename
- `BOOKMARKS_FORCE_CPU` - disable GPU (1/true/yes)
- `BOOKMARKS_GPU_LAYERS` - override layer count
- `BOOKMARKS_ENABLE_TRACING` - enable OTEL (1/true/yes)
- `PHOENIX_COLLECTOR_ENDPOINT` - OTLP endpoint for Langfuse
- `LOG_LEVEL` - log verbosity (debug/info/warn/error)

## Implementation Sequence

1. **Setup workspace** (Cargo.toml files)
2. **Common library scaffolding** (module structure, error types)
3. **Schema inference** (inference.rs, hints.rs) - pure logic, no dependencies
4. **Prompt templates** (prompt.rs) - pure logic
5. **DDL parser** (parser.rs) - pure logic with regex
6. **GPU detection** (gpu.rs) - platform-specific logic
7. **Model downloader** (downloader.rs) - hf-hub integration
8. **LLM wrapper** (model.rs) - llm crate integration
9. **Agent executor** (executor.rs) - retry loop
10. **OTEL tracing** (tracing/init.rs) - opentelemetry setup
11. **CLI** (bookmarks binary) - clap command structure
12. **Integration** - wire everything together
13. **Testing** - end-to-end schema generation test

## Files to Delete

All Python code and related files:
- `pyproject.toml`
- `src/bookmarks/` (entire Python package)
- `tests/` (Python tests)
- `native/` (PyO3 bindings - no longer needed)
- `.python-version`
- `uv.lock`
- Any `.pyc`, `__pycache__` directories

Keep:
- `README.md` (will need updates for Rust CLI)
- `DESIGN.md`
- `.gitignore` (update for Rust)
- `flake.nix` (update for Rust toolchain)
- `docker-compose.yml` (OTEL collector still relevant)

## Files to Create

All new Rust files per workspace structure above (18 new .rs files + 3 Cargo.toml files).

## Critical Design Decisions

1. **Skip candle**: Schema generation only needs LLM inference (llama.cpp), not embeddings or training
2. **Use `llm` crate**: More Rust-idiomatic than `llama-cpp-rs`, better maintained
3. **OTEL versions 0.28/0.29**: Ensures ecosystem compatibility
4. **Custom agent loop**: Simple retry logic, no framework overhead
5. **Keep same workflow**: Minimize behavior changes, port logic directly
6. **OTLP for Langfuse**: Langfuse supports OTLP ingestion natively

## Testing Strategy

Port key test cases:
- JSON schema inference with nested/array/union types
- Field hint detection (timestamps, folders)
- DDL extraction from various model outputs (with/without markdown)
- GPU detection for different platforms
- Model download and caching
- End-to-end: JSON input → SQL output

Use `cargo test` with inline test modules.
