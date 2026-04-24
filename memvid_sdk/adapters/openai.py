"""OpenAI function calling adapter for Memvid.

This adapter provides tools formatted for OpenAI's function calling API.
The tools can be used directly with the OpenAI Chat Completions API.

Usage:
    from memvid_sdk import use

    mem = use("openai", "knowledge.mv2")

    # Access tools for OpenAI function calling
    tools = mem.tools  # List of OpenAI tool definitions

    # Use with OpenAI API
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[...],
        tools=mem.tools,
        tool_choice="auto",
    )

    # Execute tool calls using mem.functions
    executors = mem.functions  # Dict of name -> callable
    for tool_call in response.choices[0].message.tool_calls:
        result = executors[tool_call.function.name](
            **json.loads(tool_call.function.arguments)
        )
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Mapping, Optional

from .._registry import registry
from .._sentinel import NoOp


def _extract_key(apikey: Optional[Mapping[str, str]]) -> Optional[str]:
    if not apikey:
        return None
    return apikey.get("openai") or apikey.get("default")


def _apply_key(key: Optional[str]) -> None:
    if not key:
        return
    try:
        import openai  # type: ignore
    except ImportError:
        warnings.warn(
            "OpenAI integration requested but 'openai' package is missing; key not applied",
            RuntimeWarning,
            stacklevel=3,
        )
        return

    openai.api_key = key  # type: ignore[attr-defined]
    os.environ.setdefault("OPENAI_API_KEY", key)


def _load_openai(
    core: Optional[Any],
    apikey: Optional[Mapping[str, str]],
) -> Mapping[str, Any]:
    key = _extract_key(apikey)
    _apply_key(key)

    if core is None:
        return {
            "tools": [],
            "functions": {},
            "nodes": NoOp("openai adapter missing core", "memvid.adapters.openai.nodes"),
            "as_query_engine": None,
        }

    # Define OpenAI-format tool definitions
    tools: List[Dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "memvid_put",
                "description": (
                    "Store a document in Memvid memory for later retrieval. "
                    "Use this to save information that should be searchable later."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Title of the document",
                        },
                        "label": {
                            "type": "string",
                            "description": "Category or label for the document",
                        },
                        "text": {
                            "type": "string",
                            "description": "Text content to store",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional key-value metadata",
                            "additionalProperties": True,
                        },
                    },
                    "required": ["title", "label", "text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memvid_find",
                "description": (
                    "Search Memvid memory for documents matching a query. "
                    "Returns the most relevant documents with snippets."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string",
                        },
                        "top_k": {
                            "type": "number",
                            "description": "Number of results to return (default: 5)",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memvid_ask",
                "description": (
                    "Ask a question and get an answer synthesized from Memvid memory "
                    "using retrieval-augmented generation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Question to answer",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["auto", "lex", "sem"],
                            "description": "Search mode: 'auto' (hybrid), 'lex' (keyword), or 'sem' (semantic)",
                        },
                    },
                    "required": ["question"],
                },
            },
        },
    ]

    # Create executor functions
    def memvid_put(
        title: str,
        label: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a document in Memvid memory."""
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

    def memvid_find(query: str, top_k: int = 5) -> str:
        """Search Memvid memory for documents."""
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
        """Ask a question and get an answer."""
        response = core.ask(question, mode=mode)
        answer = response.get("answer", "No answer generated")
        sources = response.get("sources", [])

        result = f"Answer: {answer}"
        if sources:
            source_titles = [s.get("title", "Unknown") for s in sources[:3]]
            result += f"\n\nSources: {', '.join(source_titles)}"

        return result

    # Executors dict for easy lookup
    executors = {
        "memvid_put": memvid_put,
        "memvid_find": memvid_find,
        "memvid_ask": memvid_ask,
    }

    return {
        "tools": tools,
        "functions": executors,
        "nodes": NoOp("openai adapter does not provide nodes", "memvid.adapters.openai.nodes"),
        "as_query_engine": None,
    }


registry.register("openai", _load_openai)
