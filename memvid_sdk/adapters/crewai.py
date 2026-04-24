"""CrewAI adapter exposing Memvid core methods as CrewAI tools.

CrewAI tools use either BaseTool subclasses or the @tool decorator.
This adapter provides three tools:
- memvid_put: Store documents in memory
- memvid_find: Search for relevant documents
- memvid_ask: Query with RAG-style answer synthesis

Usage:
    from memvid_sdk import use

    mem = use("crewai", "knowledge.mv2")

    # Access tools for CrewAI agents
    tools = mem.tools  # List of CrewAI Tool objects

    # Use with CrewAI Agent
    from crewai import Agent
    agent = Agent(
        role="Research Assistant",
        goal="Answer questions using stored knowledge",
        tools=mem.tools,
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Type

from .._registry import registry
from .._sentinel import NoOp
from . import _unavailable


def _load_crewai(
    core: Optional[Any],
    _apikey: Optional[Mapping[str, str]],
) -> Mapping[str, Any]:
    if core is None:
        return _unavailable("crewai", "memvid core handle missing")

    try:
        from crewai.tools import BaseTool
        from pydantic import BaseModel, Field
    except ImportError as exc:
        return _unavailable("crewai", str(exc))

    # Define input schemas using Pydantic
    class MemvidPutInput(BaseModel):
        """Input schema for storing a document in Memvid."""
        title: str = Field(..., description="Title of the document")
        label: str = Field(..., description="Category or label for the document")
        text: str = Field(..., description="Text content to store")
        metadata: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Optional key-value metadata"
        )

    class MemvidFindInput(BaseModel):
        """Input schema for searching documents in Memvid."""
        query: str = Field(..., description="Search query string")
        top_k: int = Field(default=5, description="Number of results to return")

    class MemvidAskInput(BaseModel):
        """Input schema for asking questions with RAG."""
        question: str = Field(..., description="Question to answer")
        mode: str = Field(
            default="auto",
            description="Search mode: 'auto', 'lex' (keyword), or 'sem' (semantic)"
        )

    # Define tools using BaseTool
    class MemvidPutTool(BaseTool):
        name: str = "memvid_put"
        description: str = (
            "Store a document in Memvid memory. Use this to save information "
            "that should be retrievable later. Returns the frame ID of the stored document."
        )
        args_schema: Type[BaseModel] = MemvidPutInput

        def _run(
            self,
            title: str,
            label: str,
            text: str,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> str:
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

    class MemvidFindTool(BaseTool):
        name: str = "memvid_find"
        description: str = (
            "Search Memvid memory for documents matching a query. "
            "Returns the most relevant documents with snippets and metadata."
        )
        args_schema: Type[BaseModel] = MemvidFindInput

        def _run(self, query: str, top_k: int = 5) -> str:
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

    class MemvidAskTool(BaseTool):
        name: str = "memvid_ask"
        description: str = (
            "Ask a question and get an answer synthesized from Memvid memory. "
            "Uses retrieval-augmented generation to find relevant context and generate a response."
        )
        args_schema: Type[BaseModel] = MemvidAskInput

        def _run(self, question: str, mode: str = "auto") -> str:
            response = core.ask(question, mode=mode)
            answer = response.get("answer", "No answer generated")
            sources = response.get("sources", [])

            result = f"Answer: {answer}"
            if sources:
                source_titles = [s.get("title", "Unknown") for s in sources[:3]]
                result += f"\n\nSources: {', '.join(source_titles)}"

            return result

    tools = [MemvidPutTool(), MemvidFindTool(), MemvidAskTool()]

    return {
        "tools": tools,
        "functions": [],
        "nodes": NoOp("crewai nodes not provided", "memvid.adapters.crewai.nodes"),
        "as_query_engine": None,
    }


registry.register("crewai", _load_crewai)

