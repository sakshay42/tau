from pgvector import Vector

from tau.retrieval.http import post_with_retry


def semantic_search(pg_conn, query_vector, limit=20, time_window=None):
    """Top-N documents by cosine similarity. Rows: (id, title, source, published_at, text, url, similarity).

    time_window: optional (start, end) datetimes. When given, applies a hard filter to
    published_at BETWEEN start AND end before ranking (documents with unknown published_at
    are excluded, since they can't be verified to fall inside the window).
    """
    query_vector = Vector(query_vector)

    if time_window is None:
        return pg_conn.execute(
            """
            SELECT
                id,
                title,
                source,
                published_at,
                text,
                url,
                1 - (embedding <=> %s) AS similarity
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (
                query_vector,
                query_vector,
                limit,
            )
        ).fetchall()

    start, end = time_window

    return pg_conn.execute(
        """
        SELECT
            id,
            title,
            source,
            published_at,
            text,
            url,
            1 - (embedding <=> %s) AS similarity
        FROM documents
        WHERE published_at BETWEEN %s AND %s
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (
            query_vector,
            start,
            end,
            query_vector,
            limit,
        )
    ).fetchall()


def get_document(pg_conn, document_id):
    """Fetch one document by id. Row shape matches semantic_search (similarity is NULL),
    so callers that expect a semantic_search row can use either interchangeably. Returns
    None if no document with that id exists.
    """
    return pg_conn.execute(
        """
        SELECT
            id,
            title,
            source,
            published_at,
            text,
            url,
            NULL::float AS similarity
        FROM documents
        WHERE id = %s
        """,
        (document_id,),
    ).fetchone()


def rerank_results(query, results, voyage_api_key, top_k=5):
    """Rerank semantic_search rows with Voyage rerank-2.5-lite. Returns dicts with a rerank_score."""
    documents = []

    for row in results:
        title = row[1] or ""
        text = row[4] or ""

        documents.append(
            f"{title}\n{text}".strip()
        )

    response = post_with_retry(
        "https://api.voyageai.com/v1/rerank",
        headers={
            "Authorization": f"Bearer {voyage_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "documents": documents,
            "model": "rerank-2.5-lite",
            "top_k": top_k,
        },
        timeout=60,
    )

    response.raise_for_status()

    reranked = response.json()["data"]

    final_results = []

    for item in reranked:
        row = results[item["index"]]

        final_results.append({
            "id": row[0],
            "title": row[1],
            "source": row[2],
            "published_at": row[3],
            "text": row[4],
            "url": row[5],
            "rerank_score": item["relevance_score"],
        })

    return final_results