"""AutoGen adapter exposing Memvid core methods as FunctionTool objects.

AutoGen 0.4+ uses FunctionTool to wrap Python functions for agent use.
This adapter provides three tools compatible with AutoGen's agent system:
- memvid_put: Store documents in memory
- memvid_find: Search for relevant documents
- memvid_ask: Query with RAG-style answer synthesis

Usage:
    from memvid_sdk import use

    mem = use("autogen", "knowledge.mv2")

    # Access tools for AutoGen agents
    tools = mem.tools  # List of FunctionTool objects

    # Register with an AutoGen agent
    from autogen_agentchat.agents import AssistantAgent

    agent = AssistantAgent(
        name="research_assistant",
        model_client=model_client,
        tools=mem.tools,
    )

Note: AutoGen 0.4 requires type annotations on all function parameters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional
from typing_extensions import Annotated

from .._registry import registry
from .._sentinel import NoOp
from . import _unavailable


def _load_autogen(
    core: Optional[Any],
    _apikey: Optional[Mapping[str, str]],
) -> Mapping[str, Any]:
    if core is None:
        return _unavailable("autogen", "memvid core handle missing")

    try:
        from autogen_core.tools import FunctionTool
    except ImportError:
        # Try older autogen package structure
        try:
            from autogen_core import FunctionTool
        except ImportError as exc:
            return _unavailable("autogen", str(exc))

    # Define functions with full type annotations (required by AutoGen 0.4)
    def memvid_put(
        title: Annotated[str, "Title of the document"],
        label: Annotated[str, "Category or label for the document"],
        text: Annotated[str, "Text content to store"],
        metadata: Annotated[Optional[Dict[str, Any]], "Optional key-value metadata"] = None,
    ) -> str:
        """Store a document in Memvid memory for later retrieval."""
        payload: Dict[str, Any] = {
            "title": title,
            "label": label,
            "text": text,
            "metadata": metadata or {},
            "enable_embedding": True,
            "auto_tag": True,
            "extract_dates": True,
        }
        frame_id = core.put(payload)
        return f"Document stored with frame_id: {frame_id}"

    def memvid_find(
        query: Annotated[str, "Search query string"],
        top_k: Annotated[int, "Number of results to return"] = 5,
    ) -> str:
        """Search Memvid memory for documents matching a query."""
        response = core.find(query, k=top_k)
        hits = response.get("hits", [])
        if not hits:
            return f"No results found for query: '{query}'"

        results: List[str] = []
        for i, hit in enumerate(hits, 1):
            title = hit.get("title", "Untitled")
            snippet = hit.get("text", hit.get("snippet", ""))[:200]
            score = hit.get("score", 0)
            results.append(f"{i}. [{title}] (score: {score:.2f}): {snippet}...")

        return f"Found {len(hits)} results:\n" + "\n".join(results)

    def memvid_ask(
        question: Annotated[str, "Question to answer"],
        mode: Annotated[str, "Search mode: 'auto', 'lex', or 'sem'"] = "auto",
    ) -> str:
        """Ask a question and get an answer synthesized from Memvid memory."""
        response = core.ask(question, mode=mode)
        answer = response.get("answer", "No answer generated")
        sources = response.get("sources", [])

        result = f"Answer: {answer}"
        if sources:
            source_titles = [s.get("title", "Unknown") for s in sources[:3]]
            result += f"\n\nSources: {', '.join(source_titles)}"

        return result

    # Create FunctionTool instances
    tools = [
        FunctionTool(
            memvid_put,
            description="Store a document in Memvid memory. Use this to save information that should be retrievable later.",
        ),
        FunctionTool(
            memvid_find,
            description="Search Memvid memory for documents matching a query. Returns the most relevant documents with snippets.",
        ),
        FunctionTool(
            memvid_ask,
            description="Ask a question and get an answer synthesized from Memvid memory using retrieval-augmented generation.",
        ),
    ]

    return {
        "tools": tools,
        "functions": [memvid_put, memvid_find, memvid_ask],  # Raw functions for custom usage
        "nodes": NoOp("autogen nodes not provided", "memvid.adapters.autogen.nodes"),
        "as_query_engine": None,
    }


registry.register("autogen", _load_autogen)

