"""LangChain adapter exposing Memvid core methods as tools.

LangChain tools can be created using the @tool decorator or by subclassing BaseTool.
This adapter uses the @tool decorator for simplicity and provides three tools:
- memvid_put: Store documents in memory
- memvid_find: Search for relevant documents
- memvid_ask: Query with RAG-style answer synthesis

Usage:
    from memvid_sdk import use

    mem = use("langchain", "knowledge.mv2")

    # Access tools for LangChain agents
    tools = mem.tools  # List of LangChain Tool objects

    # Use with LangChain agent
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4")
    agent = create_tool_calling_agent(llm, mem.tools, prompt)
    executor = AgentExecutor(agent=agent, tools=mem.tools)

Note: LangChain v1.x uses langchain_core.tools for the @tool decorator.
The docstring of each tool function is used as the tool description.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .._registry import registry
from .._sentinel import NoOp
from . import _unavailable


def _load_langchain(
    core: Optional[Any],
    _apikey: Optional[Mapping[str, str]],
) -> Mapping[str, Any]:
    if core is None:
        return _unavailable("langchain", "memvid core handle missing")

    try:
        from langchain_core.tools import tool
    except ImportError as exc:
        return _unavailable("langchain", str(exc))

    def _normalise_metadata(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if metadata is None:
            return {}
        return {str(key): value for key, value in metadata.items()}

    @tool("memvid_put")
    def memvid_put(
        title: str,
        label: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a document in Memvid memory for later retrieval.

        Use this tool to save information that should be searchable later.
        The document will be indexed for both keyword and semantic search.

        Args:
            title: Title of the document
            label: Category or label for the document
            text: Text content to store
            metadata: Optional key-value metadata
        """
        payload: Dict[str, Any] = {
            "title": title,
            "label": label,
            "text": text,
            "metadata": _normalise_metadata(metadata),
            "enable_embedding": True,
            "auto_tag": True,
            "extract_dates": True,
        }
        frame_id = core.put(payload)
        return f"Document stored with frame_id: {frame_id}"

    @tool("memvid_find")
    def memvid_find(query: str, top_k: int = 5) -> str:
        """Search Memvid memory for documents matching a query.

        Returns the most relevant documents with snippets and metadata.
        Use this when you need to find specific information in stored documents.

        Args:
            query: Search query string
            top_k: Number of results to return (default: 5)
        """
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

    @tool("memvid_ask")
    def memvid_ask(question: str, mode: str = "auto") -> str:
        """Ask a question and get an answer synthesized from Memvid memory.

        Uses retrieval-augmented generation to find relevant context
        and generate a comprehensive response. Best for complex questions
        that require synthesizing information from multiple sources.

        Args:
            question: Question to answer
            mode: Search mode - 'auto' (hybrid), 'lex' (keyword), or 'sem' (semantic)
        """
        response = core.ask(question, mode=mode)
        answer = response.get("answer", "No answer generated")
        sources = response.get("sources", [])

        result = f"Answer: {answer}"
        if sources:
            source_titles = [s.get("title", "Unknown") for s in sources[:3]]
            result += f"\n\nSources: {', '.join(source_titles)}"

        return result

    tools = [memvid_put, memvid_find, memvid_ask]

    return {
        "tools": tools,
        "functions": [],
        "nodes": NoOp("langchain nodes not provided", "memvid.adapters.langchain.nodes"),
        "as_query_engine": None,
    }


registry.register("langchain", _load_langchain)

