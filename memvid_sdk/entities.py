"""Entity extraction (NER) provider support for Memvid SDK.

This module provides classes for extracting named entities from text,
supporting both local NER models and cloud LLM providers.

Providers:
    - LocalNER: DistilBERT-NER (ONNX, offline, free)
    - OpenAIEntities: OpenAI GPT-4 (cloud, custom entity types)
    - ClaudeEntities: Anthropic Claude (cloud, custom entity types)
    - GeminiEntities: Google Gemini (cloud, custom entity types)
    - HuggingFaceNER: HuggingFace transformers (local)

Usage:
    from memvid_sdk import create
    from memvid_sdk.entities import get_entity_extractor, LocalNER

    # Local NER (default, no API key needed)
    ner = get_entity_extractor("local")

    # Or with cloud provider for custom entity types
    ner = get_entity_extractor("openai", entity_types=["COMPANY", "PRODUCT", "EXECUTIVE"])

    # Extract entities
    text = "Microsoft CEO Satya Nadella announced the new Surface Pro in Seattle."
    entities = ner.extract(text)
    # [
    #   {"name": "Microsoft", "type": "ORG", "confidence": 0.99},
    #   {"name": "Satya Nadella", "type": "PERSON", "confidence": 0.97},
    #   {"name": "Surface Pro", "type": "PRODUCT", "confidence": 0.95},
    #   {"name": "Seattle", "type": "LOCATION", "confidence": 0.98},
    # ]

    # Store with entities
    mem = create("knowledge.mv2", enable_lex=True)
    mem.put(title="Tech News", label="news", text=text, entities=entities)
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, TypedDict, Union


class Entity(TypedDict):
    """Extracted entity structure."""
    name: str
    type: str
    confidence: float


class EntityExtractionResult(TypedDict):
    """Result from entity extraction with optional relationships."""
    entities: List[Entity]
    relationships: Optional[List[Dict[str, Any]]]


class EntityExtractor(ABC):
    """Abstract base class for entity extraction providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name (e.g., 'local:distilbert-ner')."""
        pass

    @property
    @abstractmethod
    def entity_types(self) -> List[str]:
        """Return the supported entity types."""
        pass

    @abstractmethod
    def extract(
        self,
        text: str,
        min_confidence: float = 0.5,
    ) -> List[Entity]:
        """Extract entities from text.

        Args:
            text: Text to extract entities from
            min_confidence: Minimum confidence threshold (0.0-1.0)

        Returns:
            List of extracted entities
        """
        pass

    def extract_batch(
        self,
        texts: Sequence[str],
        min_confidence: float = 0.5,
    ) -> List[List[Entity]]:
        """Extract entities from multiple texts.

        Default implementation calls extract for each. Override for batch optimization.

        Args:
            texts: List of texts to extract entities from
            min_confidence: Minimum confidence threshold

        Returns:
            List of entity lists, one per text
        """
        return [self.extract(t, min_confidence) for t in texts]


class LocalNER(EntityExtractor):
    """Local NER provider using DistilBERT-NER (ONNX).

    Uses DistilBERT fine-tuned on CoNLL-03 for named entity recognition.
    Model files are loaded from `MEMVID_MODELS_DIR` (default: `~/.memvid/models`).
    Install them once via the CLI:
        - `memvid models install --ner distilbert-ner`

    Supported entity types (fixed):
        - PERSON: People's names
        - ORG: Organizations and companies
        - LOCATION: Places
        - MISC: Miscellaneous entities

    Example:
        >>> ner = LocalNER()
        >>> entities = ner.extract("Apple announced new products.")
        >>> # [{"name": "Apple", "type": "ORG", "confidence": 0.98}]
    """

    ENTITY_TYPES = ["PERSON", "ORG", "LOCATION", "MISC"]

    def __init__(self, model: str = "distilbert-ner"):
        """Initialize LocalNER provider.

        Args:
            model: Model to use. Default: distilbert-ner
        """
        self._model_name = model
        self._model = None

    @property
    def name(self) -> str:
        return f"local:{self._model_name}"

    @property
    def entity_types(self) -> List[str]:
        return self.ENTITY_TYPES

    def _get_model(self):
        """Lazy load the NER model from native bindings."""
        if self._model is None:
            try:
                from memvid_sdk._lib import NerModel
                self._model = NerModel()
            except ImportError as e:
                raise ImportError(
                    "Local NER support requires memvid-sdk-free built with Logic-Mesh (NER) support. "
                    "Install the model files once with `memvid models install --ner distilbert-ner` "
                    "(or set MEMVID_MODELS_DIR). "
                    f"Original error: {e}"
                )
        return self._model

    def extract(
        self,
        text: str,
        min_confidence: float = 0.5,
    ) -> List[Entity]:
        """Extract entities from text using local NER model."""
        model = self._get_model()
        raw_entities = model.extract(text)

        # Filter by confidence
        entities = [
            Entity(
                name=e["name"],
                type=e["type"],
                confidence=e["confidence"],
            )
            for e in raw_entities
            if e["confidence"] >= min_confidence
        ]

        return entities


class OpenAIEntities(EntityExtractor):
    """OpenAI GPT-4 entity extraction provider.

    Uses OpenAI's GPT models for flexible entity extraction with custom types.
    Supports relationship extraction with extract_with_relationships().

    Example:
        >>> ner = OpenAIEntities()  # Uses OPENAI_API_KEY env var
        >>> ner = OpenAIEntities(entity_types=["COMPANY", "EXECUTIVE", "PRODUCT"])
        >>> entities = ner.extract("Apple CEO Tim Cook announced iPhone 16.")
    """

    DEFAULT_ENTITY_TYPES = ["PERSON", "ORG", "LOCATION", "DATE", "PRODUCT", "EVENT", "OTHER"]

    DEFAULT_PROMPT = """Extract named entities from the provided text. Return a JSON object with an "entities" array.

Each entity should have:
- "name": The entity name as it appears in the text
- "type": The entity type from the allowed types
- "confidence": A number between 0.0 and 1.0 indicating your confidence

Return ONLY valid JSON, no explanations or markdown.

Allowed entity types: {entity_types}

Text to analyze:
{text}"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        entity_types: Optional[List[str]] = None,
        prompt: Optional[str] = None,
    ):
        """Initialize OpenAI entity extractor.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
            model: Model to use. Default: gpt-4o-mini
            entity_types: Custom entity types. Default: standard NER types
            prompt: Custom extraction prompt. Use {entity_types} and {text} placeholders.
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Pass api_key or set OPENAI_API_KEY environment variable."
            )
        self._model = model
        self._entity_types = entity_types or self.DEFAULT_ENTITY_TYPES
        self._prompt = prompt or self.DEFAULT_PROMPT
        self._client = None

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    @property
    def entity_types(self) -> List[str]:
        return self._entity_types

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package required for OpenAI entities. "
                    "Install with: pip install openai"
                )
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def extract(
        self,
        text: str,
        min_confidence: float = 0.5,
    ) -> List[Entity]:
        """Extract entities from text using OpenAI."""
        client = self._get_client()

        prompt = self._prompt.format(
            entity_types=", ".join(self._entity_types),
            text=text,
        )

        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "You are an expert at extracting named entities from text. Always return valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
        )

        content = response.choices[0].message.content or "{}"

        # Parse JSON response
        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content)
            raw_entities = data.get("entities", [])
        except json.JSONDecodeError:
            return []

        # Filter by confidence
        entities = [
            Entity(
                name=e.get("name", ""),
                type=e.get("type", "OTHER"),
                confidence=float(e.get("confidence", 0.8)),
            )
            for e in raw_entities
            if float(e.get("confidence", 0.8)) >= min_confidence
        ]

        return entities

    def extract_with_relationships(
        self,
        text: str,
        min_confidence: float = 0.5,
    ) -> EntityExtractionResult:
        """Extract entities AND relationships from text.

        Returns both entities and their relationships (e.g., "works_for", "located_in").

        Args:
            text: Text to analyze
            min_confidence: Minimum confidence threshold

        Returns:
            Dict with 'entities' and 'relationships' keys
        """
        client = self._get_client()

        prompt = f"""Extract named entities AND their relationships from the text.

Return a JSON object with:
1. "entities": array of {{"name": "...", "type": "...", "confidence": 0.9}}
2. "relationships": array of {{"source": "entity name", "target": "entity name", "type": "RELATIONSHIP_TYPE", "confidence": 0.9}}

Entity types: {", ".join(self._entity_types)}
Relationship types: WORKS_FOR, LOCATED_IN, OWNS, PRODUCES, PARTNER_OF, SUBSIDIARY_OF, CEO_OF, FOUNDED, ACQUIRED, OTHER

Return ONLY valid JSON.

Text: {text}"""

        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "You are an expert at extracting entities and relationships. Always return valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2000,
        )

        content = response.choices[0].message.content or "{}"

        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content)
        except json.JSONDecodeError:
            return EntityExtractionResult(entities=[], relationships=[])

        entities = [
            Entity(
                name=e.get("name", ""),
                type=e.get("type", "OTHER"),
                confidence=float(e.get("confidence", 0.8)),
            )
            for e in data.get("entities", [])
            if float(e.get("confidence", 0.8)) >= min_confidence
        ]

        relationships = [
            r for r in data.get("relationships", [])
            if float(r.get("confidence", 0.8)) >= min_confidence
        ]

        return EntityExtractionResult(entities=entities, relationships=relationships)


class ClaudeEntities(EntityExtractor):
    """Anthropic Claude entity extraction provider.

    Uses Claude for sophisticated entity extraction, especially good for
    complex documents and nuanced text.

    Example:
        >>> ner = ClaudeEntities()  # Uses ANTHROPIC_API_KEY env var
        >>> entities = ner.extract(complex_legal_document)
    """

    DEFAULT_ENTITY_TYPES = ["PERSON", "ORG", "LOCATION", "DATE", "PRODUCT", "EVENT", "OTHER"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        entity_types: Optional[List[str]] = None,
    ):
        """Initialize Claude entity extractor.

        Args:
            api_key: Anthropic API key. If not provided, uses ANTHROPIC_API_KEY env var.
            model: Model to use. Default: claude-3-5-sonnet-20241022
            entity_types: Custom entity types.
        """
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Anthropic API key required. Pass api_key or set ANTHROPIC_API_KEY environment variable."
            )
        self._model = model
        self._entity_types = entity_types or self.DEFAULT_ENTITY_TYPES
        self._client = None

    @property
    def name(self) -> str:
        return f"claude:{self._model}"

    @property
    def entity_types(self) -> List[str]:
        return self._entity_types

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required for Claude entities. "
                    "Install with: pip install anthropic"
                )
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def extract(
        self,
        text: str,
        min_confidence: float = 0.5,
    ) -> List[Entity]:
        """Extract entities from text using Claude."""
        client = self._get_client()

        prompt = f"""Extract named entities from this text. Return ONLY a JSON object with an "entities" array.

Each entity: {{"name": "exact text", "type": "TYPE", "confidence": 0.9}}
Types: {", ".join(self._entity_types)}

Text: {text}"""

        response = client.messages.create(
            model=self._model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text if response.content else "{}"

        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content)
            raw_entities = data.get("entities", [])
        except json.JSONDecodeError:
            return []

        entities = [
            Entity(
                name=e.get("name", ""),
                type=e.get("type", "OTHER"),
                confidence=float(e.get("confidence", 0.8)),
            )
            for e in raw_entities
            if float(e.get("confidence", 0.8)) >= min_confidence
        ]

        return entities


class GeminiEntities(EntityExtractor):
    """Google Gemini entity extraction provider.

    Uses Gemini for fast, accurate entity extraction.

    Example:
        >>> ner = GeminiEntities()  # Uses GEMINI_API_KEY env var
        >>> entities = ner.extract("Google announced Gemini 2.0 in Mountain View.")
    """

    DEFAULT_ENTITY_TYPES = ["PERSON", "ORG", "LOCATION", "DATE", "PRODUCT", "EVENT", "OTHER"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        entity_types: Optional[List[str]] = None,
    ):
        """Initialize Gemini entity extractor.

        Args:
            api_key: Google AI API key. If not provided, uses GEMINI_API_KEY env var.
            model: Model to use. Default: gemini-2.0-flash
            entity_types: Custom entity types.
        """
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Gemini API key required. Pass api_key or set GEMINI_API_KEY environment variable."
            )
        self._model = model
        self._entity_types = entity_types or self.DEFAULT_ENTITY_TYPES
        self._client = None

    @property
    def name(self) -> str:
        return f"gemini:{self._model}"

    @property
    def entity_types(self) -> List[str]:
        return self._entity_types

    def _get_client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "google-generativeai package required for Gemini entities. "
                    "Install with: pip install google-generativeai"
                )
            genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(self._model)
        return self._client

    def extract(
        self,
        text: str,
        min_confidence: float = 0.5,
    ) -> List[Entity]:
        """Extract entities from text using Gemini."""
        model = self._get_client()

        prompt = f"""Extract named entities from this text. Return ONLY a JSON object with an "entities" array.

Each entity: {{"name": "exact text", "type": "TYPE", "confidence": 0.9}}
Types: {", ".join(self._entity_types)}

Text: {text}"""

        response = model.generate_content(prompt)
        content = response.text

        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content)
            raw_entities = data.get("entities", [])
        except json.JSONDecodeError:
            return []

        entities = [
            Entity(
                name=e.get("name", ""),
                type=e.get("type", "OTHER"),
                confidence=float(e.get("confidence", 0.8)),
            )
            for e in raw_entities
            if float(e.get("confidence", 0.8)) >= min_confidence
        ]

        return entities


class HuggingFaceNER(EntityExtractor):
    """HuggingFace NER provider using transformers.

    Runs NER models locally via HuggingFace transformers.

    Example:
        >>> ner = HuggingFaceNER()  # Uses dslim/bert-base-NER
        >>> ner = HuggingFaceNER(model="dslim/distilbert-NER")  # Faster
        >>> entities = ner.extract("Apple announced new products.")
    """

    # Map BIO tags to entity types
    TAG_MAP = {
        "PER": "PERSON",
        "LOC": "LOCATION",
        "ORG": "ORG",
        "MISC": "MISC",
    }

    def __init__(self, model: str = "dslim/bert-base-NER"):
        """Initialize HuggingFace NER provider.

        Args:
            model: HuggingFace model name. Options:
                   - dslim/bert-base-NER (default)
                   - dslim/distilbert-NER (faster)
                   - Jean-Baptiste/roberta-large-ner-english (more accurate)
        """
        self._model_name = model
        self._pipeline = None

    @property
    def name(self) -> str:
        return f"huggingface:{self._model_name}"

    @property
    def entity_types(self) -> List[str]:
        return ["PERSON", "ORG", "LOCATION", "MISC"]

    def _get_pipeline(self):
        """Lazy load the NER pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError:
                raise ImportError(
                    "transformers package required for HuggingFace NER. "
                    "Install with: pip install transformers torch"
                )
            self._pipeline = pipeline(
                "ner",
                model=self._model_name,
                aggregation_strategy="simple",
            )
        return self._pipeline

    def extract(
        self,
        text: str,
        min_confidence: float = 0.5,
    ) -> List[Entity]:
        """Extract entities from text using HuggingFace NER."""
        pipe = self._get_pipeline()
        results = pipe(text)

        entities = []
        for r in results:
            # Map entity group to standard type
            entity_group = r.get("entity_group", "").replace("B-", "").replace("I-", "")
            entity_type = self.TAG_MAP.get(entity_group, entity_group)

            confidence = float(r.get("score", 0.0))
            if confidence >= min_confidence:
                entities.append(Entity(
                    name=r.get("word", "").strip(),
                    type=entity_type,
                    confidence=confidence,
                ))

        return entities


def _parse_provider_model(spec: str) -> tuple[str, Optional[str]]:
    """Parse provider:model specification.

    Examples:
        "openai" -> ("openai", None)
        "openai:gpt-4o-mini" -> ("openai", "gpt-4o-mini")
        "claude:claude-3-5-sonnet-20241022" -> ("claude", "claude-3-5-sonnet-20241022")
        "gemini:gemini-2.0-flash" -> ("gemini", "gemini-2.0-flash")
    """
    if ":" in spec:
        parts = spec.split(":", 1)
        return parts[0].lower(), parts[1]
    return spec.lower(), None


def get_entity_extractor(
    provider: str = "local",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    entity_types: Optional[List[str]] = None,
    **kwargs,
) -> EntityExtractor:
    """Factory function to create an entity extraction provider.

    Supports two formats:
        1. Separate provider and model: get_entity_extractor("openai", model="gpt-4o")
        2. Combined format: get_entity_extractor("openai:gpt-4o")

    Args:
        provider: Provider name or "provider:model" format.
                  Supported: local, openai, claude, gemini, huggingface
        model: Model name (uses provider default if not specified)
        api_key: API key for cloud providers
        entity_types: Custom entity types (cloud providers only)
        **kwargs: Additional provider-specific arguments

    Returns:
        EntityExtractor instance

    Examples:
        >>> ner = get_entity_extractor("local")
        >>> ner = get_entity_extractor("openai")
        >>> ner = get_entity_extractor("openai:gpt-4o")
        >>> ner = get_entity_extractor("openai:gpt-4o-mini", entity_types=["COMPANY", "PRODUCT"])
        >>> ner = get_entity_extractor("claude:claude-3-5-sonnet-20241022")
        >>> ner = get_entity_extractor("gemini:gemini-2.0-flash")
    """
    # Parse provider:model format
    parsed_provider, parsed_model = _parse_provider_model(provider)

    # Use parsed model if model not explicitly provided
    if model is None:
        model = parsed_model

    provider = parsed_provider

    if provider == "local":
        return LocalNER(model=model or "distilbert-ner")
    elif provider == "openai":
        return OpenAIEntities(
            api_key=api_key,
            model=model or "gpt-4o-mini",
            entity_types=entity_types,
            **kwargs,
        )
    elif provider in ("claude", "anthropic"):
        return ClaudeEntities(
            api_key=api_key,
            model=model or "claude-3-5-sonnet-20241022",
            entity_types=entity_types,
            **kwargs,
        )
    elif provider == "gemini":
        return GeminiEntities(
            api_key=api_key,
            model=model or "gemini-2.0-flash",
            entity_types=entity_types,
            **kwargs,
        )
    elif provider in ("huggingface", "hf"):
        return HuggingFaceNER(model=model or "dslim/bert-base-NER")
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: local, openai, claude, gemini, huggingface"
        )


__all__ = [
    "Entity",
    "EntityExtractionResult",
    "EntityExtractor",
    "LocalNER",
    "OpenAIEntities",
    "ClaudeEntities",
    "GeminiEntities",
    "HuggingFaceNER",
    "get_entity_extractor",
]
