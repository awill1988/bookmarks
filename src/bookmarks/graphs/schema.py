import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, TypedDict

from langchain_community.llms.llamacpp import LlamaCpp
from langgraph.graph import END, StateGraph

from bookmarks.models.gpu import get_gpu_config

DEFAULT_SCHEMA_SQL_PATH = Path("schema.sql")


class SchemaGraphState(TypedDict, total=False):
    source_path: Path
    output_path: Path
    raw_payload: Any
    json_schema: dict[str, Any]
    field_hints: dict[str, list[str]]
    sql_text: str


def _infer_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def _merge_types(lhs: str | list[str], rhs: str | list[str]) -> list[str]:
    merged: set[str] = set()
    merged.update([lhs] if isinstance(lhs, str) else lhs)
    merged.update([rhs] if isinstance(rhs, str) else rhs)
    return sorted(merged)


def _merge_schema(lhs: dict[str, Any], rhs: dict[str, Any]) -> dict[str, Any]:
    result = dict(lhs)
    if "type" in lhs and "type" in rhs:
        result["type"] = _merge_types(lhs["type"], rhs["type"])
    elif "type" in rhs:
        result["type"] = rhs["type"]

    if lhs.get("type") == "object" and rhs.get("type") == "object":
        props: dict[str, Any] = {**lhs.get("properties", {})}
        for key, value in rhs.get("properties", {}).items():
            if key in props:
                props[key] = _merge_schema(props[key], value)
            else:
                props[key] = value
        required = set(lhs.get("required", [])) | set(rhs.get("required", []))
        result["properties"] = props
        if required:
            result["required"] = sorted(required)

    if lhs.get("type") == "array" and rhs.get("type") == "array":
        if "items" in lhs and "items" in rhs:
            result["items"] = _merge_schema(lhs["items"], rhs["items"])
        else:
            result["items"] = lhs.get("items") or rhs.get("items")

    examples: list[Any] = []
    if "examples" in lhs:
        examples.extend(lhs["examples"])
    if "examples" in rhs:
        examples.extend(rhs["examples"])
    if examples:
        result["examples"] = examples[:3]

    return result


def infer_json_schema(payload: Any) -> dict[str, Any]:
    """Lightweight heuristic JSON schema inference for bookmark exports."""

    def visit(node: Any) -> dict[str, Any]:
        node_type = _infer_type(node)

        if node_type == "object":
            properties: dict[str, Any] = {}
            required_keys: set[str] = set()
            for key, value in node.items():
                value_schema = visit(value)
                if key in properties:
                    properties[key] = _merge_schema(properties[key], value_schema)
                else:
                    properties[key] = value_schema
                if value is not None:
                    required_keys.add(key)
            schema: dict[str, Any] = {"type": "object", "properties": properties}
            if required_keys:
                schema["required"] = sorted(required_keys)
            return schema

        if node_type == "array":
            items_schema: dict[str, Any] | None = None
            for value in node:
                value_schema = visit(value)
                if items_schema is None:
                    items_schema = value_schema
                else:
                    items_schema = _merge_schema(items_schema, value_schema)
            schema = {"type": "array"}
            if items_schema is not None:
                schema["items"] = items_schema
            return schema

        schema = {"type": node_type}
        if node is not None:
            schema["examples"] = [node]
        return schema

    return visit(payload)


def derive_field_hints(json_schema: dict[str, Any]) -> dict[str, list[str]]:
    """Collect field names that look like timestamps or folder metadata."""

    timestamp_keys: set[str] = set()
    folder_keys: set[str] = set()

    def walk(schema: dict[str, Any], path: list[str]) -> None:
        schema_type = schema.get("type")
        if schema_type == "object":
            for key, value in (schema.get("properties") or {}).items():
                lowered = key.lower()
                if any(token in lowered for token in ("date", "time", "updated", "created")):
                    timestamp_keys.add(".".join(path + [key]))
                if any(token in lowered for token in ("folder", "path", "group", "category", "tag")):
                    folder_keys.add(".".join(path + [key]))
                walk(value, path + [key])
        if schema_type == "array" and "items" in schema:
            walk(schema["items"], path + ["[]"])

    walk(json_schema, [])
    return {"timestamps": sorted(timestamp_keys), "folders": sorted(folder_keys)}


def build_schema_prompt_from_data(json_schema: dict[str, Any], field_hints: dict[str, list[str]]) -> str:
    schema_text = json.dumps(json_schema, indent=2)
    timestamps = field_hints.get("timestamps") or []
    folders = field_hints.get("folders") or []
    hint_lines = []
    if timestamps:
        hint_lines.append(f"timestamp candidates: {', '.join(timestamps)}")
    if folders:
        hint_lines.append(f"folder/tag candidates: {', '.join(folders)}")
    hints = "\n".join(hint_lines) if hint_lines else "no explicit timestamp or folder hints found"

    return (
        "You are generating a deterministic sqlite schema for bookmarks. "
        "Use the provided JSON Schema to identify meaningful columns, including url, title, "
        "folder or path metadata, timestamps, and any labels/tags present. "
        "Return a single CREATE TABLE statement that captures those columns with reasonable "
        "types and constraints. Avoid INSERT statements. "
        f"\n\nJSON Schema:\n{schema_text}\n\nHints:\n{hints}"
    )


def load_raw_payload_node(state: SchemaGraphState) -> SchemaGraphState:
    source_path = Path(state["source_path"])
    logging.info("loading bookmarks from %s", source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"bookmark source {source_path} does not exist")

    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    logging.info("loaded %d top-level items", len(payload) if isinstance(payload, list) else 1)
    return {"raw_payload": payload}


def infer_schema_node(state: SchemaGraphState) -> SchemaGraphState:
    payload = state.get("raw_payload")
    if payload is None:
        raise ValueError("schema graph requires raw_payload")

    logging.info("inferring json schema from payload structure")
    json_schema = infer_json_schema(payload)
    field_hints = derive_field_hints(json_schema)

    hint_summary = []
    if field_hints.get("timestamps"):
        hint_summary.append(f"{len(field_hints['timestamps'])} timestamp fields")
    if field_hints.get("folders"):
        hint_summary.append(f"{len(field_hints['folders'])} folder/tag fields")
    if hint_summary:
        logging.info("detected: %s", ", ".join(hint_summary))

    return {"json_schema": json_schema, "field_hints": field_hints}


def synthesize_sql_node(model: LlamaCpp):
    def _synthesize(state: SchemaGraphState) -> SchemaGraphState:
        json_schema = state.get("json_schema") or {}
        field_hints = state.get("field_hints") or {}
        prompt = build_schema_prompt_from_data(json_schema, field_hints)
        logging.info("invoking llm to generate sql schema (this may take a while)")
        try:
            result = model.invoke(prompt)
        except Exception as exc:  # pragma: no cover - passthrough for runtime issues
            raise RuntimeError(f"schema synthesis failed: {exc}") from exc
        logging.info("llm synthesis complete")
        return {"sql_text": str(result).strip()}

    return _synthesize


def persist_sql_node(default_output: Path | None = None):
    def _persist(state: SchemaGraphState) -> SchemaGraphState:
        output_path = Path(state.get("output_path") or default_output or DEFAULT_SCHEMA_SQL_PATH)
        sql_text = state.get("sql_text")
        if not sql_text:
            raise ValueError("no sql_text available for persistence")
        logging.info("writing sql schema to %s", output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(sql_text, encoding="utf-8")
        return {"output_path": output_path}

    return _persist


def build_schema_graph(
    model_path: Path | str,
    output_path: Path | None = None,
    llm_factory: Callable[..., LlamaCpp] = LlamaCpp,
):
    """Create a LangGraph pipeline to infer schema and synthesize SQL via a local model."""
    gpu_config = get_gpu_config()

    logging.info("loading language model from %s", model_path)
    if gpu_config.is_accelerated:
        logging.info("gpu acceleration enabled: %s backend", gpu_config.backend)

    model = llm_factory(
        model_path=str(model_path),
        temperature=0.0,
        n_ctx=4096,  # context window size - needs to handle large bookmark schemas
        n_batch=512,  # batch size for prompt processing
        n_threads=max(os.cpu_count() or 1, 1),
        n_gpu_layers=gpu_config.n_gpu_layers,  # offload layers to GPU if available
        verbose=True,  # show llama.cpp loading progress
    )
    logging.info("model loaded successfully")
    graph = StateGraph(SchemaGraphState)

    graph.add_node("load", load_raw_payload_node)
    graph.add_node("infer_schema", infer_schema_node)
    graph.add_node("synthesize_sql", synthesize_sql_node(model))
    graph.add_node("persist_sql", persist_sql_node(output_path))

    graph.set_entry_point("load")
    graph.add_edge("load", "infer_schema")
    graph.add_edge("infer_schema", "synthesize_sql")
    graph.add_edge("synthesize_sql", "persist_sql")
    graph.add_edge("persist_sql", END)

    return graph.compile()
