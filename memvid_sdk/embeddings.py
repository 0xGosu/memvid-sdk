"""External embedding provider support for Memvid SDK.

This module provides classes for generating embeddings using external providers
like OpenAI, allowing users to use their own embedding models with Memvid.

Usage:
    from memvid_sdk import create
    from memvid_sdk.embeddings import OpenAIEmbeddings

    # Initialize embedding provider
    embedder = OpenAIEmbeddings(api_key="sk-...")  # or uses OPENAI_API_KEY env var

    # Create memory with external embeddings
    mem = create("knowledge.mv2", enable_vec=True)

    # Store documents with embeddings
    docs = [
        {"title": "Doc 1", "label": "notes", "text": "Content 1..."},
        {"title": "Doc 2", "label": "notes", "text": "Content 2..."},
    ]
    embeddings = embedder.embed_documents([d["text"] for d in docs])
    mem.put_many(docs, embeddings=embeddings)

    # Query with embedding
    query = "search query"
    query_embedding = embedder.embed_query(query)
    results = mem.find(query, query_embedding=query_embedding)
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from json import dumps, loads
from typing import List, Optional, Sequence


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension for this model."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/identifier."""
        pass

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents.

        Args:
            texts: List of text documents to embed

        Returns:
            List of embedding vectors, one per document
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector for the query
        """
        pass


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embedding provider.

    Uses OpenAI's text-embedding models to generate embeddings.
    Compatible with text-embedding-3-small (1536 dim), text-embedding-3-large (3072 dim),
    and text-embedding-ada-002 (1536 dim).

    Example:
        >>> embedder = OpenAIEmbeddings()  # Uses OPENAI_API_KEY env var
        >>> embedder = OpenAIEmbeddings(api_key="sk-...", model="text-embedding-3-large")
        >>> vectors = embedder.embed_documents(["Hello world", "Goodbye world"])
        >>> query_vec = embedder.embed_query("Hello")
    """

    # Model dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
    ):
        """Initialize OpenAI embedding provider.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
            model: Model to use. One of: text-embedding-3-small, text-embedding-3-large,
                   text-embedding-ada-002
            batch_size: Number of texts to embed in a single API call
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Pass api_key or set OPENAI_API_KEY environment variable."
            )
        self._model = model
        self._batch_size = batch_size
        self._client: Optional["openai.OpenAI"] = None

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package required for OpenAI embeddings. "
                    "Install with: pip install openai"
                )
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    @property
    def dimension(self) -> int:
        """Return embedding dimension for the current model."""
        return self.MODEL_DIMENSIONS.get(self._model, 1536)

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate embeddings for documents using OpenAI API.

        Args:
            texts: List of text documents to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        client = self._get_client()
        all_embeddings: List[List[float]] = []

        # Process in batches
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = client.embeddings.create(
                model=self._model,
                input=list(batch),
            )
            # Sort by index to ensure correct order
            sorted_embeddings = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([e.embedding for e in sorted_embeddings])

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        client = self._get_client()
        response = client.embeddings.create(
            model=self._model,
            input=text,
        )
        return response.data[0].embedding


class CohereEmbeddings(EmbeddingProvider):
    """Cohere embedding provider.

    Uses Cohere's embedding models.

    Example:
        >>> embedder = CohereEmbeddings()  # Uses COHERE_API_KEY env var
        >>> vectors = embedder.embed_documents(["Hello world"])
    """

    MODEL_DIMENSIONS = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-light-v3.0": 384,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "embed-english-v3.0",
        input_type: str = "search_document",
    ):
        """Initialize Cohere embedding provider.

        Args:
            api_key: Cohere API key. If not provided, uses COHERE_API_KEY env var.
            model: Model to use
            input_type: One of: search_document, search_query, classification, clustering
        """
        self._api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Cohere API key required. Pass api_key or set COHERE_API_KEY environment variable."
            )
        self._model = model
        self._input_type = input_type
        self._client = None

    def _get_client(self):
        """Lazy initialization of Cohere client."""
        if self._client is None:
            try:
                import cohere
            except ImportError:
                raise ImportError(
                    "cohere package required. Install with: pip install cohere"
                )
            self._client = cohere.Client(api_key=self._api_key)
        return self._client

    @property
    def dimension(self) -> int:
        return self.MODEL_DIMENSIONS.get(self._model, 1024)

    @property
    def model_name(self) -> str:
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate embeddings for documents."""
        if not texts:
            return []
        client = self._get_client()
        response = client.embed(
            texts=list(texts),
            model=self._model,
            input_type=self._input_type,
        )
        return [list(e) for e in response.embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query."""
        client = self._get_client()
        response = client.embed(
            texts=[text],
            model=self._model,
            input_type="search_query",
        )
        return list(response.embeddings[0])


class VoyageEmbeddings(EmbeddingProvider):
    """Voyage AI embedding provider.

    Uses Voyage AI's embedding models, known for high-quality embeddings.

    Example:
        >>> embedder = VoyageEmbeddings()  # Uses VOYAGE_API_KEY env var
        >>> vectors = embedder.embed_documents(["Hello world"])
    """

    MODEL_DIMENSIONS = {
        "voyage-3": 1024,
        "voyage-3-lite": 512,
        "voyage-code-3": 1024,
        "voyage-finance-2": 1024,
        "voyage-law-2": 1024,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "voyage-3",
    ):
        """Initialize Voyage embedding provider.

        Args:
            api_key: Voyage API key. If not provided, uses VOYAGE_API_KEY env var.
            model: Model to use
        """
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Voyage API key required. Pass api_key or set VOYAGE_API_KEY environment variable."
            )
        self._model = model
        self._client = None

    def _get_client(self):
        """Lazy initialization of Voyage client."""
        if self._client is None:
            try:
                import voyageai
            except ImportError:
                raise ImportError(
                    "voyageai package required. Install with: pip install voyageai"
                )
            self._client = voyageai.Client(api_key=self._api_key)
        return self._client

    @property
    def dimension(self) -> int:
        return self.MODEL_DIMENSIONS.get(self._model, 1024)

    @property
    def model_name(self) -> str:
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate embeddings for documents."""
        if not texts:
            return []
        client = self._get_client()
        result = client.embed(list(texts), model=self._model, input_type="document")
        return [list(e) for e in result.embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query."""
        client = self._get_client()
        result = client.embed([text], model=self._model, input_type="query")
        return list(result.embeddings[0])


class NvidiaEmbeddings(EmbeddingProvider):
    """NVIDIA embedding provider.

    Uses NVIDIA Integrate API to generate embeddings.

    Example:
        >>> from memvid_sdk.embeddings import NvidiaEmbeddings
        >>> embedder = NvidiaEmbeddings()  # Uses NVIDIA_API_KEY env var
        >>> vec = embedder.embed_query("What is the capital of France?")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "nvidia/nv-embed-v1",
        batch_size: int = 64,
        document_input_type: str = "passage",
        query_input_type: str = "query",
        encoding_format: str = "float",
        truncate: str = "NONE",
        dimension: Optional[int] = None,
    ):
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self._api_key:
            raise ValueError(
                "NVIDIA API key required. Pass api_key or set NVIDIA_API_KEY environment variable."
            )
        base_url_candidate = (
            base_url
            or os.environ.get("NVIDIA_BASE_URL")
            or "https://integrate.api.nvidia.com"
        ).strip()
        self._base_url = (
            base_url_candidate if base_url_candidate else "https://integrate.api.nvidia.com"
        ).rstrip("/")

        env_model = os.environ.get("NVIDIA_EMBEDDING_MODEL")
        model_candidate = (env_model if env_model is not None else model).strip()
        self._model = model_candidate if model_candidate else "nvidia/nv-embed-v1"
        self._batch_size = int(batch_size)
        self._document_input_type = document_input_type
        self._query_input_type = query_input_type
        self._encoding_format = encoding_format
        self._truncate = truncate
        self._dimension = int(dimension) if dimension else None

    @property
    def dimension(self) -> int:
        return self._dimension or 0

    @property
    def model_name(self) -> str:
        return self._model

    def _set_dimension_from_embedding(self, embedding: Sequence[float]) -> None:
        if self._dimension is None and embedding:
            self._dimension = len(embedding)

    def _post(self, payload: dict) -> dict:
        url = f"{self._base_url}/v1/embeddings"
        body = dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"NVIDIA API error: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"NVIDIA API error: {exc}") from exc
        return loads(data.decode("utf-8", errors="replace"))

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = list(texts[i : i + self._batch_size])
            data = self._post(
                {
                    "input": batch,
                    "model": self._model,
                    "input_type": self._document_input_type,
                    "encoding_format": self._encoding_format,
                    "truncate": self._truncate,
                }
            )
            entries = data.get("data") or []
            entries = sorted(entries, key=lambda item: item.get("index", 0))
            batch_embeddings: List[List[float]] = []
            for item in entries:
                embedding = item.get("embedding")
                if not isinstance(embedding, list) or not embedding:
                    raise RuntimeError("NVIDIA API error: empty embedding returned")
                batch_embeddings.append(embedding)

            if batch_embeddings:
                self._set_dimension_from_embedding(batch_embeddings[0])
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        data = self._post(
            {
                "input": [text],
                "model": self._model,
                "input_type": self._query_input_type,
                "encoding_format": self._encoding_format,
                "truncate": self._truncate,
            }
        )
        embedding = ((data.get("data") or [{}])[0] or {}).get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError("NVIDIA API error: empty embedding returned")
        self._set_dimension_from_embedding(embedding)
        return embedding


class GeminiEmbeddings(EmbeddingProvider):
    """Gemini (Google AI) embedding provider.

    Uses Google's Gemini API to generate embeddings.
    Compatible with text-embedding-004 (768d) and gemini-embedding-001 (3072d).

    Example:
        >>> from memvid_sdk.embeddings import GeminiEmbeddings
        >>> embedder = GeminiEmbeddings()  # Uses GOOGLE_API_KEY or GEMINI_API_KEY
        >>> embedder = GeminiEmbeddings(model="gemini-embedding-001")
        >>> vec = embedder.embed_query("What is the capital of France?")
    """

    MODEL_DIMENSIONS = {
        "text-embedding-004": 768,
        "gemini-embedding-001": 3072,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-004",
        task_type: Optional[str] = None,
    ):
        """Initialize Gemini embedding provider.

        Args:
            api_key: Google API key. If not provided, uses GOOGLE_API_KEY or GEMINI_API_KEY env var.
            model: Model to use. Default: text-embedding-004.
            task_type: Task type hint for embeddings. Options:
                       RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT, SEMANTIC_SIMILARITY,
                       CLASSIFICATION, CLUSTERING. Auto-set if not specified.
        """
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Google API key required. Pass api_key or set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )
        self._model = model
        self._task_type = task_type

    @property
    def dimension(self) -> int:
        return self.MODEL_DIMENSIONS.get(self._model, 768)

    @property
    def model_name(self) -> str:
        return self._model

    def _post(self, url: str, payload: dict) -> dict:
        body = dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API error: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Gemini API error: {exc}") from exc
        return loads(data.decode("utf-8", errors="replace"))

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        if not texts:
            return []

        # Gemini supports batch embedding
        requests = [
            {
                "model": f"models/{self._model}",
                "content": {"parts": [{"text": text}]},
                "taskType": self._task_type or "RETRIEVAL_DOCUMENT",
            }
            for text in texts
        ]

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:batchEmbedContents?key={self._api_key}"
        data = self._post(url, {"requests": requests})
        embeddings = data.get("embeddings", [])
        return [e.get("values", []) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:embedContent?key={self._api_key}"
        payload = {
            "model": f"models/{self._model}",
            "content": {"parts": [{"text": text}]},
            "taskType": self._task_type or "RETRIEVAL_QUERY",
        }
        data = self._post(url, payload)
        return data.get("embedding", {}).get("values", [])


class MistralEmbeddings(EmbeddingProvider):
    """Mistral AI embedding provider.

    Uses Mistral's embeddings API to generate high-quality embeddings.
    Uses mistral-embed model (1024 dimensions).

    Example:
        >>> from memvid_sdk.embeddings import MistralEmbeddings
        >>> embedder = MistralEmbeddings()  # Uses MISTRAL_API_KEY
        >>> vectors = embedder.embed_documents(["Hello world"])
    """

    DIMENSION = 1024

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistral-embed",
        batch_size: int = 100,
    ):
        """Initialize Mistral embedding provider.

        Args:
            api_key: Mistral API key. If not provided, uses MISTRAL_API_KEY env var.
            model: Model to use. Default: mistral-embed.
            batch_size: Number of texts to embed per API call. Default: 100.
        """
        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Mistral API key required. Pass api_key or set MISTRAL_API_KEY environment variable."
            )
        self._model = model
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    @property
    def model_name(self) -> str:
        return self._model

    def _post(self, payload: dict) -> dict:
        url = "https://api.mistral.ai/v1/embeddings"
        body = dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Mistral API error: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Mistral API error: {exc}") from exc
        return loads(data.decode("utf-8", errors="replace"))

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = list(texts[i : i + self._batch_size])
            data = self._post({"model": self._model, "input": batch})
            # Sort by index to maintain order
            items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
            all_embeddings.extend([item.get("embedding", []) for item in items])

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query."""
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []


class OllamaEmbeddings(EmbeddingProvider):
    """Ollama local embedding provider.

    Uses a local Ollama server to generate embeddings. No API key required.
    Supports models like nomic-embed-text, mxbai-embed-large, all-minilm, etc.

    Example:
        >>> embedder = OllamaEmbeddings()  # Uses nomic-embed-text by default
        >>> embedder = OllamaEmbeddings(model="mxbai-embed-large")
        >>> vectors = embedder.embed_documents(["Hello world"])
        >>> query_vec = embedder.embed_query("Hello")
    """

    # Known model dimensions for popular Ollama embedding models
    MODEL_DIMENSIONS = {
        # General purpose
        "nomic-embed-text": 768,
        "nomic-embed-text:v1": 768,
        "nomic-embed-text:v1.5": 768,
        "mxbai-embed-large": 1024,
        "mxbai-embed-large:v1": 1024,
        "all-minilm": 384,
        "all-minilm:l6-v2": 384,
        "all-minilm:l12-v2": 384,
        # Snowflake Arctic (various sizes)
        "snowflake-arctic-embed": 1024,
        "snowflake-arctic-embed:s": 384,
        "snowflake-arctic-embed:m": 768,
        "snowflake-arctic-embed:l": 1024,
        "snowflake-arctic-embed:335m": 1024,
        # BGE models
        "bge-m3": 1024,
        "bge-large": 1024,
        "bge-large:en": 1024,
        "bge-large:en-v1.5": 1024,
        "bge-base": 768,
        "bge-base:en": 768,
        "bge-base:en-v1.5": 768,
        "bge-small": 384,
        "bge-small:en": 384,
        "bge-small:en-v1.5": 384,
        # Jina embeddings
        "jina-embeddings-v2-base-en": 768,
        "jina-embeddings-v2-small-en": 512,
        # Multilingual
        "paraphrase-multilingual": 768,
        "paraphrase-multilingual:mpnet-base-v2": 768,
        # E5 models
        "e5-large": 1024,
        "e5-base": 768,
        "e5-small": 384,
        "e5-mistral-7b-instruct": 4096,
    }

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        """Initialize Ollama embedding provider.

        Args:
            model: Model to use. Default: nomic-embed-text.
                   Popular options: nomic-embed-text (768d), mxbai-embed-large (1024d),
                   all-minilm (384d), snowflake-arctic-embed (1024d).
            base_url: Ollama server URL. Default: http://localhost:11434 or OLLAMA_HOST env var.
            dimension: Override embedding dimension. Auto-detected from model if not specified.
        """
        default_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        base_url_candidate = (base_url or default_host).strip().rstrip("/")
        self._base_url = base_url_candidate if base_url_candidate else "http://localhost:11434"
        self._model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        if self._dimension:
            return self._dimension
        return self.MODEL_DIMENSIONS.get(self._model, 768)

    @property
    def model_name(self) -> str:
        return self._model

    def _post(self, payload: dict) -> dict:
        url = f"{self._base_url}/api/embeddings"
        body = dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama API error: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama API error: {exc}. Is Ollama running at {self._base_url}?"
            ) from exc
        return loads(data.decode("utf-8", errors="replace"))

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents.

        Note: Ollama API embeds one text at a time, so this processes sequentially.
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for text in texts:
            data = self._post({"model": self._model, "prompt": text})
            embedding = data.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise RuntimeError("Ollama API error: empty embedding returned")
            # Auto-detect dimension from first embedding
            if self._dimension is None and embedding:
                self._dimension = len(embedding)
            all_embeddings.append(embedding)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        data = self._post({"model": self._model, "prompt": text})
        embedding = data.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError("Ollama API error: empty embedding returned")
        # Auto-detect dimension
        if self._dimension is None and embedding:
            self._dimension = len(embedding)
        return embedding


class HuggingFaceEmbeddings(EmbeddingProvider):
    """HuggingFace local embedding provider using sentence-transformers.

    Runs models locally for privacy and no API costs.

    Example:
        >>> embedder = HuggingFaceEmbeddings()  # Uses all-MiniLM-L6-v2 by default
        >>> embedder = HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")
        >>> vectors = embedder.embed_documents(["Hello world"])
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """Initialize HuggingFace embedding provider.

        Args:
            model: Model name or path. Popular options:
                   - all-MiniLM-L6-v2 (384 dim, fast)
                   - BAAI/bge-small-en-v1.5 (384 dim)
                   - BAAI/bge-base-en-v1.5 (768 dim)
                   - sentence-transformers/all-mpnet-base-v2 (768 dim)
            device: Device to use (cpu, cuda, mps). Auto-detected if not specified.
        """
        self._model_name = model
        self._device = device
        self._model = None
        self._dimension: Optional[int] = None

    def _get_model(self):
        """Lazy initialization of sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers package required. "
                    "Install with: pip install sentence-transformers"
                )
            self._model = SentenceTransformer(self._model_name, device=self._device)
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._get_model()
        return self._dimension or 384

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate embeddings for documents locally."""
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(list(texts), convert_to_numpy=True)
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query locally."""
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


class HashEmbeddings(EmbeddingProvider):
    """Deterministic offline embedding provider.

    Intended for tests and offline demos where you need stable embeddings without
    network access or model downloads.
    """

    def __init__(self, dimension: int = 32, model: str = "memvid-hash-32"):
        if dimension <= 0:
            raise ValueError("dimension must be > 0")
        self._dimension = int(dimension)
        self._model = str(model)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    def _embed_one(self, text: str) -> List[float]:
        import hashlib

        data = text.encode("utf-8", errors="replace")
        digest = hashlib.sha256(data).digest()
        out: List[float] = []
        for i in range(self._dimension):
            out.append(digest[i % len(digest)] / 255.0)
        return out

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)


# Convenience function for common use cases
def get_embedder(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> EmbeddingProvider:
    """Factory function to create an embedding provider.

    Args:
        provider: One of: openai, cohere, voyage, huggingface
        model: Model name (uses provider default if not specified)
        api_key: API key (uses environment variable if not specified)
        **kwargs: Additional provider-specific arguments

    Returns:
        EmbeddingProvider instance

    Example:
        >>> embedder = get_embedder("openai")
        >>> embedder = get_embedder("huggingface", model="BAAI/bge-small-en-v1.5")
    """
    provider = provider.lower()

    if provider in ("hash", "deterministic"):
        return HashEmbeddings(
            dimension=int(kwargs.pop("dimension", 32)),
            model=model or "memvid-hash-32",
        )
    elif provider == "openai":
        return OpenAIEmbeddings(
            api_key=api_key,
            model=model or "text-embedding-3-small",
            **kwargs,
        )
    elif provider == "cohere":
        return CohereEmbeddings(
            api_key=api_key,
            model=model or "embed-english-v3.0",
            **kwargs,
        )
    elif provider == "voyage":
        return VoyageEmbeddings(
            api_key=api_key,
            model=model or "voyage-3",
            **kwargs,
        )
    elif provider == "nvidia":
        return NvidiaEmbeddings(
            api_key=api_key,
            model=model or "nvidia/nv-embed-v1",
            **kwargs,
        )
    elif provider in ("gemini", "google"):
        return GeminiEmbeddings(
            api_key=api_key,
            model=model or "text-embedding-004",
            **kwargs,
        )
    elif provider == "mistral":
        return MistralEmbeddings(
            api_key=api_key,
            model=model or "mistral-embed",
            **kwargs,
        )
    elif provider in ("huggingface", "hf", "local"):
        return HuggingFaceEmbeddings(
            model=model or "all-MiniLM-L6-v2",
            **kwargs,
        )
    elif provider == "ollama":
        return OllamaEmbeddings(
            model=model or "nomic-embed-text",
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: hash, openai, cohere, voyage, nvidia, gemini, mistral, huggingface, ollama"
        )


# Local embedding models available via fastembed (no API key required).
#
# These run locally using ONNX runtime. Use via the `embedding_model` parameter:
#
# Example:
#     # Local embeddings - no API key needed
#     mem.put(
#         title="Doc",
#         label="docs",
#         metadata={},
#         text="Content...",
#         enable_embedding=True,
#         embedding_model="bge-small",  # 384d, fast, good quality
#     )
#
#     # Or use constants
#     from memvid_sdk.embeddings import LOCAL_EMBEDDING_MODELS
#     mem.put(
#         title="Doc",
#         label="docs",
#         metadata={},
#         text="Content...",
#         enable_embedding=True,
#         embedding_model=LOCAL_EMBEDDING_MODELS["BGE_SMALL"],
#     )
#
# First run may download model weights to MEMVID_MODELS_DIR.

LOCAL_EMBEDDING_MODELS = {
    # BGE-small-en-v1.5 (384 dimensions) - Fast, good quality
    "BGE_SMALL": "bge-small",
    # BGE-base-en-v1.5 (768 dimensions) - Balanced speed/quality
    "BGE_BASE": "bge-base",
    # Nomic-embed-text-v1.5 (768 dimensions) - Good general purpose
    "NOMIC": "nomic",
    # GTE-large-en-v1.5 (1024 dimensions) - Highest quality, slower
    "GTE_LARGE": "gte-large",
}

LOCAL_MODEL_DIMENSIONS = {
    "bge-small": 384,
    "bge-base": 768,
    "nomic": 768,
    "gte-large": 1024,
}


__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbeddings",
    "CohereEmbeddings",
    "VoyageEmbeddings",
    "NvidiaEmbeddings",
    "GeminiEmbeddings",
    "MistralEmbeddings",
    "OllamaEmbeddings",
    "HuggingFaceEmbeddings",
    "HashEmbeddings",
    "get_embedder",
    "LOCAL_EMBEDDING_MODELS",
    "LOCAL_MODEL_DIMENSIONS",
]
