"""MCP tool server exposing tau's retrieval primitives.

MCP is just an interface layer here — every tool below is a thin wrapper around
functions already in `tau.retrieval.search` / `tau.retrieval.embeddings`. No SQL
or Voyage API logic is duplicated; this module only adds typed tool schemas and
JSON-safe serialization (Postgres rows -> dicts, datetimes -> ISO-8601 strings)
around calls that already exist.

Three tools only, matching what the temporal-routing agent's three routes need:

- search_semantic: unrestricted semantic search       (current / topical routes)
- search_recent:   semantic search + hard time filter  (explicit_temporal route)
- get_document:    look up one stored document by id
"""

from datetime import datetime
from typing import Optional

from mcp.server.mcpserver import MCPServer

from tau.retrieval.embeddings import embed_query
from tau.retrieval.search import get_document as _get_document
from tau.retrieval.search import semantic_search


def _row_to_dict(row):
    """A semantic_search/get_document row -> a JSON-safe dict.

    Row shape: (id, title, source, published_at, text, url, similarity).
    """
    doc_id, title, source, published_at, text, url, similarity = row

    return {
        "id": doc_id,
        "title": title,
        "source": source,
        "published_at": published_at.isoformat() if published_at else None,
        "text": text,
        "url": url,
        "similarity": similarity,
    }


def build_mcp_server(pg_conn, voyage_api_key) -> MCPServer:
    """Build the MCP server, wired to the same Postgres connection and Voyage
    API key the rest of the retrieval pipeline already uses.
    """

    server = MCPServer(
        name="tau-retrieval",
        instructions=(
            "Retrieval tools over tau's ingested and embedded document corpus. "
            "Use search_recent for queries with an explicit time window, "
            "search_semantic for everything else, and get_document to fetch a "
            "specific document by id."
        ),
    )

    @server.tool()
    def search_semantic(query: str, k: int = 20) -> list[dict]:
        """Semantic top-k search over the whole corpus (pgvector cosine similarity).

        Use for queries with no explicit time constraint.
        """
        query_vector = embed_query(query, voyage_api_key)
        rows = semantic_search(pg_conn, query_vector, limit=k)
        return [_row_to_dict(row) for row in rows]

    @server.tool()
    def search_recent(query: str, start_time: str, end_time: str, k: int = 20) -> list[dict]:
        """Semantic top-k search hard-filtered to published_at in [start_time, end_time].

        start_time / end_time are ISO-8601 datetimes. Use for explicit temporal queries
        (e.g. "in the last 3 hours") where the extracted window is a hard constraint,
        not just a ranking signal.
        """
        query_vector = embed_query(query, voyage_api_key)
        window = (datetime.fromisoformat(start_time), datetime.fromisoformat(end_time))
        rows = semantic_search(pg_conn, query_vector, limit=k, time_window=window)
        return [_row_to_dict(row) for row in rows]

    @server.tool()
    def get_document(document_id: str) -> Optional[dict]:
        """Look up one stored document by id. Returns None if it doesn't exist."""
        row = _get_document(pg_conn, document_id)
        return _row_to_dict(row) if row else None

    return server
