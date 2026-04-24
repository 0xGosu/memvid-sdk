"""Google ADK (Agent Development Kit) adapter for Memvid.

This adapter provides tools formatted for Google's Agent Development Kit.
The tools can be used directly with ADK agents and the Gemini API.

Usage:
    from memvid_sdk import use

    mem = use("google-adk", "knowledge.mv2")

    # Access tools for Google GenAI SDK
    tools = mem.tools  # types.Tool object with FunctionDeclarations

    # Use with Google GenAI SDK
    from google import genai
    from google.genai import types

    client = genai.Client()
    chat = client.chats.create(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            tools=[mem.tools],
            system_instruction="You are a helpful assistant."
        )
    )

    # Execute tool calls using mem.functions
    executors = mem.functions  # Dict of name -> callable
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
    return apikey.get("google") or apikey.get("default")


def _apply_key(key: Optional[str]) -> None:
    if not key:
        return
    os.environ.setdefault("GOOGLE_API_KEY", key)


def _build_tools() -> Any:
    """Build Google GenAI SDK tools if available, otherwise return dict format."""
    # Define function schemas as dicts (portable format)
    function_schemas: List[Dict[str, Any]] = [
        {
            "name": "memvid_put",
            "description": (
                "Store a document in Memvid memory for later retrieval. "
                "Use this to save information that should be searchable later."
            ),
            "parameters_json_schema": {
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
                    },
                },
                "required": ["title", "label", "text"],
            },
        },
        {
            "name": "memvid_find",
            "description": (
                "Search Memvid memory for documents matching a query. "
                "Returns the most relevant documents with snippets and scores."
            ),
            "parameters_json_schema": {
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
        {
            "name": "memvid_ask",
            "description": (
                "Ask a question and get an answer synthesized from Memvid memory "
                "using retrieval-augmented generation."
            ),
            "parameters_json_schema": {
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
    ]

    # Try to build proper google-genai types
    try:
        from google.genai import types

        function_declarations = []
        for schema in function_schemas:
            func_decl = types.FunctionDeclaration(
                name=schema["name"],
                description=schema["description"],
                parameters_json_schema=schema["parameters_json_schema"],
            )
            function_declarations.append(func_decl)

        return types.Tool(function_declarations=function_declarations)
    except ImportError:
        # Fall back to dict format for other uses
        return function_schemas


def _load_google_adk(
    core: Optional[Any],
    apikey: Optional[Mapping[str, str]],
) -> Mapping[str, Any]:
    key = _extract_key(apikey)
    _apply_key(key)

    if core is None:
        return {
            "tools": [],
            "functions": {},
            "nodes": NoOp("google-adk adapter missing core", "memvid.adapters.google-adk.nodes"),
            "as_query_engine": None,
        }

    # Build tools (returns types.Tool if google-genai is available, else list of dicts)
    tools = _build_tools()

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
        "nodes": NoOp("google-adk adapter does not provide nodes", "memvid.adapters.google-adk.nodes"),
        "as_query_engine": None,
    }


registry.register("google-adk", _load_google_adk)
