import json
import logging
import re
from typing import Any

from openinference.semconv.trace import SpanAttributes
from smolagents.models import ChatMessage, MessageRole, Model

from bookmarks.tracing import get_tracer

SCHEMA_SYSTEM_PROMPT = (
    "you generate deterministic sqlite ddl for bookmarks. "
    "output only a single create table statement. "
    "no markdown, no commentary, no backticks, no multiple tables."
)


def _extract_create_table(sql_text: str) -> str:
    text = sql_text.strip()
    if not text:
        raise ValueError("model returned empty output")

    fenced = re.search(r"```(?:sql)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    match = re.search(r"create\s+table\b", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError("model output did not contain create table")

    ddl = text[match.start() :].strip()
    ddl = ddl.split("\n\n", 1)[0].strip()
    return ddl


def build_schema_task_prompt(
    json_schema: dict[str, Any],
    field_hints: dict[str, list[str]],
) -> str:
    schema_text = json.dumps(json_schema, indent=2, ensure_ascii=False)
    timestamps = field_hints.get("timestamps") or []
    folders = field_hints.get("folders") or []

    hint_lines = []
    if timestamps:
        hint_lines.append(f"timestamp candidates: {', '.join(timestamps)}")
    if folders:
        hint_lines.append(f"folder/tag candidates: {', '.join(folders)}")
    hints = "\n".join(hint_lines) if hint_lines else "no explicit timestamp or folder hints found"

    return (
        "generate a single sqlite create table statement for bookmarks.\n"
        "- include url and title columns.\n"
        "- include timestamp columns if present in the export.\n"
        "- include folder/path metadata if present.\n"
        "- include an embeddings vector column.\n"
        "- keep types and constraints reasonable and deterministic.\n"
        "- output only the create table statement.\n\n"
        f"json schema:\n{schema_text}\n\nhints:\n{hints}"
    )


def generate_schema_sql(model: Model, prompt: str) -> str:
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("schema.generate_sql") as span:
        span.set_attribute(SpanAttributes.INPUT_VALUE, prompt)
        message = model.generate(
            [
                ChatMessage(role=MessageRole.SYSTEM, content=SCHEMA_SYSTEM_PROMPT),
                ChatMessage(role=MessageRole.USER, content=prompt),
            ],
            stop_sequences=None,
        )
        output = str(message.content or "").strip()
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, output)

    return _extract_create_table(output)


def generate_schema_sql_with_retry(model: Model, prompt: str, *, max_attempts: int = 2) -> str:
    last_error: Exception | None = None
    current_prompt = prompt

    for attempt in range(1, max_attempts + 1):
        try:
            return generate_schema_sql(model, current_prompt)
        except Exception as exc:
            last_error = exc
            logging.warning("schema synthesis attempt %s failed: %s", attempt, exc)
            current_prompt = (
                f"{prompt}\n\nprevious output was invalid: {exc}\n"
                "return only one sqlite create table statement. no notes."
            )

    raise RuntimeError(f"schema synthesis failed after {max_attempts} attempts: {last_error}") from last_error

