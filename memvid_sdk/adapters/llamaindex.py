"""LlamaIndex adapter exposing Memvid helpers as FunctionTool and QueryEngineTool.

LlamaIndex uses FunctionTool for wrapping Python functions and QueryEngineTool
for wrapping query engines. This adapter provides:
- memvid_put: Store documents in memory
- memvid_find: Search for relevant documents
- memvid_ask: Query with RAG-style answer synthesis
- as_query_engine(): Factory for creating a LlamaIndex QueryEngine

Usage:
    from memvid_sdk import use

    mem = use("llamaindex", "knowledge.mv2")

    # Access tools for LlamaIndex agents
    tools = mem.tools  # List of FunctionTool objects

    # Use with LlamaIndex agent
    from llama_index.core.agent import ReActAgent
    from llama_index.llms.openai import OpenAI

    llm = OpenAI(model="gpt-4")
    agent = ReActAgent.from_tools(mem.tools, llm=llm)
    response = agent.chat("What do you know about X?")

    # Or use as a QueryEngine
    query_engine = mem.as_query_engine()
    response = query_engine.query("Summarize the key points")
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .._registry import registry
from .._sentinel import NoOp
from . import _unavailable


def _load_llamaindex(
    core: Optional[Any],
    _apikey: Optional[Mapping[str, str]],
) -> Mapping[str, Any]:
    if core is None:
        return _unavailable("llamaindex", "memvid core handle missing")

    try:
        from llama_index.core.tools import FunctionTool
    except ImportError as exc:
        return _unavailable("llamaindex", str(exc))

    def memvid_put(
        title: str,
        label: str,
        text: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store a document in Memvid memory for later retrieval.

        Args:
            title: Title of the document
            label: Category or label for the document
            text: Text content to store
            metadata: Optional key-value metadata
        """
        payload = {
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

    def memvid_find(query: str, top_k: int = 5) -> str:
        """Search Memvid memory for documents matching a query.

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

    def memvid_ask(question: str, mode: str = "auto") -> str:
        """Ask a question and get an answer synthesized from Memvid memory.

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

    tools = [
        FunctionTool.from_defaults(
            fn=memvid_put,
            name="memvid_put",
            description=(
                "Store a document in Memvid memory. Use this to save information "
                "that should be retrievable later."
            ),
        ),
        FunctionTool.from_defaults(
            fn=memvid_find,
            name="memvid_find",
            description=(
                "Search Memvid memory for documents matching a query. "
                "Returns the most relevant documents with snippets."
            ),
        ),
        FunctionTool.from_defaults(
            fn=memvid_ask,
            name="memvid_ask",
            description=(
                "Ask a question and get an answer synthesized from Memvid memory "
                "using retrieval-augmented generation."
            ),
        ),
    ]

    def _query_engine_factory():
        """Create a LlamaIndex QueryEngine backed by Memvid."""
        try:
            from llama_index.core.query_engine import CustomQueryEngine
            from llama_index.core import Response
        except ImportError:
            return None

        class MemvidQueryEngine(CustomQueryEngine):
            """QueryEngine implementation that uses Memvid for retrieval and RAG."""

            # Store core reference as class attribute
            _memvid_core: Any = None

            def __init__(self, memvid_core: Any, **kwargs):
                super().__init__(**kwargs)
                self._memvid_core = memvid_core

            def custom_query(self, query_str: str) -> Response:
                result = self._memvid_core.ask(query_str)
                answer = result.get("answer", "No answer generated")
                sources = result.get("sources", [])

                # Build source nodes for the response
                source_text = ""
                if sources:
                    source_titles = [s.get("title", "Unknown") for s in sources[:3]]
                    source_text = f"\n\nSources: {', '.join(source_titles)}"

                return Response(response=answer + source_text)

        return MemvidQueryEngine(memvid_core=core)

    return {
        "tools": tools,
        "functions": [memvid_put, memvid_find, memvid_ask],
        "nodes": NoOp("llamaindex nodes not provided", "memvid.adapters.llamaindex.nodes"),
        "as_query_engine": _query_engine_factory,
    }


registry.register("llamaindex", _load_llamaindex)

