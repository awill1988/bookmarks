"""Compatibility exports for bookmark workflows."""

from bookmarks.graphs.embed import (
    DEFAULT_DB_PATH,
    GraphState,
    build_demo_graph,
    embed_bookmarks_node,
    load_bookmarks_node,
    normalize_payload,
    persist_sqlite_node,
)
from bookmarks.graphs.schema import (
    DEFAULT_SCHEMA_SQL_PATH,
    SchemaGraphState,
    build_schema_graph,
    build_schema_prompt_from_data,
    derive_field_hints,
    infer_json_schema,
    infer_schema_node,
    load_raw_payload_node,
    persist_sql_node,
    synthesize_sql_node,
)

__all__ = [
    "DEFAULT_DB_PATH",
    "GraphState",
    "build_demo_graph",
    "embed_bookmarks_node",
    "load_bookmarks_node",
    "normalize_payload",
    "persist_sqlite_node",
    "DEFAULT_SCHEMA_SQL_PATH",
    "SchemaGraphState",
    "build_schema_graph",
    "build_schema_prompt_from_data",
    "derive_field_hints",
    "infer_json_schema",
    "infer_schema_node",
    "load_raw_payload_node",
    "persist_sql_node",
    "synthesize_sql_node",
]
