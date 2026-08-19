"""Minimal LangGraph wrapper around tau's temporal-routed retrieval pipeline.

This module does not re-implement any retrieval logic. Every node is a thin
wrapper around the existing functions in `tau.retrieval.*` — the graph's only
job is to sequence them:

    query -> classify intent -> retrieve (path depends on intent)
          -> rerank -> [tau decay, current route only] -> final_results

The three routes match `tau.retrieval.router.retrieve_with_temporal_routing`
exactly, just expressed as explicit graph steps instead of one function body:

- explicit_temporal: hard time-window filter -> semantic search -> rerank. No decay.
- current:           semantic search -> rerank -> tau decay.
- topical:           semantic search -> rerank. No decay.

Two graph builders are provided:

- `build_temporal_agent_graph` — retrieval calls `tau.retrieval.search` directly
  (used by notebook 02/03).
- `build_temporal_agent_graph_mcp` — the same graph, except the `retrieve` node
  goes through the MCP tools in `tau.mcp.server` (`search_semantic` /
  `search_recent`) instead of calling `semantic_search`/`embed_query` directly
  (used by notebook 04). Reranking and tau decay are not exposed as MCP tools
  (out of scope), so those two nodes are unchanged.
"""

from datetime import datetime
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from tau.retrieval.embeddings import embed_query
from tau.retrieval.router import classify_query, extract_time_window
from tau.retrieval.search import rerank_results, semantic_search
from tau.retrieval.temporal import apply_temporal_decay


class TemporalAgentState(TypedDict, total=False):
    query: str
    route: str
    time_window: Optional[tuple]
    results: list
    reranked_results: list
    tau_results: list
    final_results: list
    mcp_tool: str
    mcp_arguments: dict


def build_temporal_agent_graph(
    pg_conn,
    voyage_api_key,
    limit=20,
    top_k=5,
    tau_hours=24,
    now=None,
):
    """Compile the temporal-routing graph.

    `pg_conn` and `voyage_api_key` are captured by the node closures (they're
    external resources, not part of the state that flows through the graph).
    """

    def classify_intent(state: TemporalAgentState) -> dict:
        query = state["query"]
        route = classify_query(query)
        time_window = (
            extract_time_window(query, now=now) if route == "explicit_temporal" else None
        )
        return {"route": route, "time_window": time_window}

    def retrieve(state: TemporalAgentState) -> dict:
        query_vector = embed_query(state["query"], voyage_api_key)
        results = semantic_search(
            pg_conn,
            query_vector,
            limit=limit,
            time_window=state.get("time_window"),
        )
        return {"results": results}

    def rerank(state: TemporalAgentState) -> dict:
        reranked_results = rerank_results(
            state["query"], state["results"], voyage_api_key, top_k=top_k
        )
        return {"reranked_results": reranked_results}

    def apply_tau(state: TemporalAgentState) -> dict:
        tau_results = apply_temporal_decay(
            state["reranked_results"], tau_hours=tau_hours, now=now
        )
        return {"tau_results": tau_results, "final_results": tau_results}

    def finalize(state: TemporalAgentState) -> dict:
        # explicit_temporal / topical: the reranked order already is the final
        # order — no tau decay applies.
        return {"final_results": state["reranked_results"]}

    def after_rerank(state: TemporalAgentState) -> str:
        return "apply_tau" if state["route"] == "current" else "finalize"

    graph = StateGraph(TemporalAgentState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("apply_tau", apply_tau)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "classify_intent")
    graph.add_edge("classify_intent", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges(
        "rerank", after_rerank, {"apply_tau": "apply_tau", "finalize": "finalize"}
    )
    graph.add_edge("apply_tau", END)
    graph.add_edge("finalize", END)

    return graph.compile()


def _mcp_row_to_tuple(document: dict) -> tuple:
    """An MCP search_semantic/search_recent result dict -> a semantic_search-shaped
    row, so rerank_results (unchanged) can consume it the same way it consumes a
    direct Postgres row: (id, title, source, published_at, text, url).
    """
    published_at = document["published_at"]

    return (
        document["id"],
        document["title"],
        document["source"],
        datetime.fromisoformat(published_at) if published_at else None,
        document["text"],
        document["url"],
    )


def build_temporal_agent_graph_mcp(
    mcp_client,
    voyage_api_key,
    limit=20,
    top_k=5,
    tau_hours=24,
    now=None,
):
    """Compile the temporal-routing graph with retrieval routed through MCP tools.

    Identical to `build_temporal_agent_graph` except the `retrieve` node calls the
    `search_semantic` / `search_recent` MCP tools (via an already-connected
    `mcp.Client`) instead of `semantic_search`/`embed_query` directly. Routing
    (`classify_intent`), reranking, and tau decay are untouched — those aren't MCP
    tools, so they still call `tau.retrieval.*` directly.
    """

    def classify_intent(state: TemporalAgentState) -> dict:
        query = state["query"]
        route = classify_query(query)
        time_window = (
            extract_time_window(query, now=now) if route == "explicit_temporal" else None
        )
        return {"route": route, "time_window": time_window}

    async def retrieve(state: TemporalAgentState) -> dict:
        query = state["query"]
        time_window = state.get("time_window")

        if time_window is not None:
            start, end = time_window
            tool_name = "search_recent"
            arguments = {
                "query": query,
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "k": limit,
            }
        else:
            tool_name = "search_semantic"
            arguments = {"query": query, "k": limit}

        tool_result = await mcp_client.call_tool(tool_name, arguments)

        if tool_result.is_error:
            raise RuntimeError(f"MCP tool {tool_name!r} failed: {tool_result.content}")

        documents = tool_result.structured_content["result"]
        results = [_mcp_row_to_tuple(doc) for doc in documents]

        return {"results": results, "mcp_tool": tool_name, "mcp_arguments": arguments}

    def rerank(state: TemporalAgentState) -> dict:
        reranked_results = rerank_results(
            state["query"], state["results"], voyage_api_key, top_k=top_k
        )
        return {"reranked_results": reranked_results}

    def apply_tau(state: TemporalAgentState) -> dict:
        tau_results = apply_temporal_decay(
            state["reranked_results"], tau_hours=tau_hours, now=now
        )
        return {"tau_results": tau_results, "final_results": tau_results}

    def finalize(state: TemporalAgentState) -> dict:
        return {"final_results": state["reranked_results"]}

    def after_rerank(state: TemporalAgentState) -> str:
        return "apply_tau" if state["route"] == "current" else "finalize"

    graph = StateGraph(TemporalAgentState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("apply_tau", apply_tau)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "classify_intent")
    graph.add_edge("classify_intent", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges(
        "rerank", after_rerank, {"apply_tau": "apply_tau", "finalize": "finalize"}
    )
    graph.add_edge("apply_tau", END)
    graph.add_edge("finalize", END)

    return graph.compile()
