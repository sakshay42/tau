import re
from datetime import datetime, timedelta, timezone

from tau.retrieval.embeddings import embed_query
from tau.retrieval.search import semantic_search, rerank_results
from tau.retrieval.temporal import apply_temporal_decay


def classify_query(query):
    q = query.lower()

    explicit_patterns = [
        r"last \d+ hours?",
        r"last \d+ days?",
        r"today",
        r"yesterday",
        r"this morning",
        r"this afternoon",
        r"this evening",
        r"this week",
    ]

    for pattern in explicit_patterns:
        if re.search(pattern, q):
            return "explicit_temporal"

    current_words = [
        "latest",
        "recent",
        "currently",
        "what's happening",
        "what is happening",
        "what's going on",
        "what is going on",
        "current",
        "news",
        "updates",
    ]

    if any(word in q for word in current_words):
        return "current"

    return "topical"



def extract_time_window(query, now=None):
    if now is None:
        now = datetime.now(timezone.utc)

    q = query.lower()

    match = re.search(r"last (\d+) hours?", q)
    if match:
        hours = int(match.group(1))
        return now - timedelta(hours=hours), now

    match = re.search(r"last (\d+) days?", q)
    if match:
        days = int(match.group(1))
        return now - timedelta(days=days), now

    if "today" in q:
        start = now.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        return start, now

    if "yesterday" in q:
        today = now.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        return today - timedelta(days=1), today

    return None


def retrieve_with_temporal_routing(
    query,
    pg_conn,
    voyage_api_key,
    limit=20,
    top_k=5,
    tau_hours=24,
    now=None,
):
    """Classify a query's temporal intent and run the matching retrieval pipeline.

    - explicit_temporal: hard filter to the extracted time window, semantic search inside
      it, rerank. No soft decay (the hard filter already is the temporal constraint).
      If a window can't be extracted despite the explicit-temporal classification, falls
      back to an unfiltered search (behaves like "topical").
    - current: semantic search, rerank, then soft tau decay by recency.
    - topical: semantic search, rerank. No temporal adjustment.

    Returns a dict: {"query", "intent", "time_window", "results"}, where "results" is
    reranked_results (explicit_temporal / topical) or tau_results (current).
    """
    intent = classify_query(query)
    query_vector = embed_query(query, voyage_api_key)

    time_window = None

    if intent == "explicit_temporal":
        time_window = extract_time_window(query, now=now)

    results = semantic_search(pg_conn, query_vector, limit=limit, time_window=time_window)
    reranked_results = rerank_results(query, results, voyage_api_key, top_k=top_k)

    if intent == "current":
        final_results = apply_temporal_decay(reranked_results, tau_hours=tau_hours, now=now)
    else:
        final_results = reranked_results

    return {
        "query": query,
        "intent": intent,
        "time_window": time_window,
        "results": final_results,
    }