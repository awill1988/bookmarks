use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "bookmarks")]
#[command(about = "bookmark schema generation tool", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate database schema from bookmark JSON
    Gen {
        #[command(subcommand)]
        subcommand: GenCommands,
    },
}

#[derive(Subcommand)]
enum GenCommands {
    /// Generate SQL schema from bookmark JSON
    Schema {
        /// Input JSON file path
        #[arg(short, long)]
        input: PathBuf,

        /// Output SQL file path
        #[arg(short, long)]
        output: PathBuf,

        /// HuggingFace model repository ID
        #[arg(long, env = "BOOKMARKS_SCHEMA_REPO_ID")]
        repo_id: Option<String>,

        /// GGUF model filename
        #[arg(long, env = "BOOKMARKS_SCHEMA_FILENAME")]
        filename: Option<String>,

        /// Model cache directory
        #[arg(long, env = "BOOKMARKS_MODEL_CACHE_DIR")]
        cache_dir: Option<PathBuf>,

        /// Maximum retry attempts for schema generation
        #[arg(long, default_value = "2")]
        max_attempts: usize,
    },
}

impl Cli {
    pub async fn execute(self) -> Result<()> {
        match self.command {
            Commands::Gen { subcommand } => match subcommand {
                GenCommands::Schema {
                    input,
                    output,
                    repo_id,
                    filename,
                    cache_dir,
                    max_attempts,
                } => {
                    generate_schema(input, output, repo_id, filename, cache_dir, max_attempts)
                        .await
                }
            },
        }
    }
}

async fn generate_schema(
    input: PathBuf,
    output: PathBuf,
    repo_id: Option<String>,
    filename: Option<String>,
    cache_dir: Option<PathBuf>,
    max_attempts: usize,
) -> Result<()> {
    use common::agent::{build_schema_task_prompt, generate_schema_sql_with_retry};
    use common::llm::{ensure_gguf_model, LlamaModel, ModelConfig};
    use common::schema::{derive_field_hints, infer_json_schema};
    use common::tracing::init_tracing;

    // initialize tracing
    let _guard = init_tracing("bookmarks")?;

    tracing::info!("loading bookmarks from {}", input.display());

    // load JSON file
    let json_text = std::fs::read_to_string(&input)?;
    let payload: serde_json::Value = serde_json::from_str(&json_text)?;

    let item_count = if payload.is_array() {
        payload.as_array().map(|a| a.len()).unwrap_or(0)
    } else {
        1
    };
    tracing::info!("loaded {} top-level items", item_count);

    // infer schema
    tracing::info!("inferring json schema from payload structure");
    let schema = infer_json_schema(&payload);
    let hints = derive_field_hints(&schema);

    // log detected hints
    let mut hint_summary = Vec::new();
    if !hints.timestamps.is_empty() {
        hint_summary.push(format!("{} timestamp fields", hints.timestamps.len()));
    }
    if !hints.folders.is_empty() {
        hint_summary.push(format!("{} folder/tag fields", hints.folders.len()));
    }
    if !hint_summary.is_empty() {
        tracing::info!("detected: {}", hint_summary.join(", "));
    }

    // download/load model
    tracing::info!("ensuring model is available");
    let model_path = ensure_gguf_model(repo_id, filename, cache_dir)?;

    tracing::info!("loading language model from {}", model_path.display());
    let model = LlamaModel::new(model_path, ModelConfig::default())?;
    tracing::info!("model loaded successfully");

    // build prompt
    let prompt = build_schema_task_prompt(&schema, &hints);

    // generate SQL with retry
    tracing::info!("invoking llm to generate sql schema (this may take a while)");
    let sql_text = generate_schema_sql_with_retry(&model, &prompt, max_attempts)?;
    tracing::info!("llm synthesis complete");

    // write output
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&output, &sql_text)?;

    tracing::info!("wrote sql schema to {}", output.display());

    Ok(())
}
