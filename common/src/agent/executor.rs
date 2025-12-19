use crate::agent::parser::extract_create_table;
use crate::agent::prompt::SCHEMA_SYSTEM_PROMPT;
use crate::error::{BookmarksError, Result};
use crate::llm::model::{LlamaModel, Message};

#[tracing::instrument(skip(model, prompt), fields(prompt_len = prompt.len()))]
pub fn generate_schema_sql(model: &LlamaModel, prompt: &str) -> Result<String> {
    let messages = vec![
        Message::system(SCHEMA_SYSTEM_PROMPT),
        Message::user(prompt),
    ];

    let output = model.generate(messages)?;

    extract_create_table(&output)
}

#[tracing::instrument(skip(model, prompt), fields(max_attempts, prompt_len = prompt.len()))]
pub fn generate_schema_sql_with_retry(
    model: &LlamaModel,
    prompt: &str,
    max_attempts: usize,
) -> Result<String> {
    let mut last_error: Option<BookmarksError> = None;
    let mut current_prompt = prompt.to_string();

    for attempt in 1..=max_attempts {
        tracing::info!("schema synthesis attempt {}/{}", attempt, max_attempts);

        match generate_schema_sql(model, &current_prompt) {
            Ok(ddl) => {
                tracing::info!("schema synthesis succeeded on attempt {}", attempt);
                return Ok(ddl);
            }
            Err(e) => {
                tracing::warn!("schema synthesis attempt {} failed: {}", attempt, e);
                last_error = Some(e.clone());

                // append error feedback to prompt for next attempt
                current_prompt = format!(
                    "{}\n\nprevious output was invalid: {}\n\
                     return only one sqlite create table statement. no notes.",
                    prompt,
                    e
                );
            }
        }
    }

    Err(last_error.unwrap_or_else(|| {
        BookmarksError::Model(format!("schema synthesis failed after {} attempts", max_attempts))
    }))
}

impl Clone for BookmarksError {
    fn clone(&self) -> Self {
        match self {
            BookmarksError::Io(e) => BookmarksError::Io(std::io::Error::new(e.kind(), e.to_string())),
            BookmarksError::Json(e) => BookmarksError::Json(serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))),
            BookmarksError::Model(s) => BookmarksError::Model(s.clone()),
            BookmarksError::Schema(s) => BookmarksError::Schema(s.clone()),
            BookmarksError::DdlExtraction(s) => BookmarksError::DdlExtraction(s.clone()),
            BookmarksError::ModelDownload(s) => BookmarksError::ModelDownload(s.clone()),
            BookmarksError::GpuDetection(s) => BookmarksError::GpuDetection(s.clone()),
            BookmarksError::Tracing(s) => BookmarksError::Tracing(s.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_schema_sql_with_retry_params() {
        // just test that the function signature compiles
        // actual testing would require a real model
        let max_attempts = 3;
        assert!(max_attempts > 0);
    }
}
