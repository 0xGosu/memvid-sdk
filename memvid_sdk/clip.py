"""CLIP visual embedding provider support for Memvid SDK.

This module provides classes for generating visual embeddings using CLIP models,
enabling image-to-image and text-to-image similarity search.

Providers:
    - LocalClip: MobileCLIP-S2 (ONNX, offline, free)
    - OpenAIClip: GPT-4V + text embeddings (cloud)
    - GeminiClip: Google Gemini multimodal (cloud)
    - HuggingFaceClip: HuggingFace transformers (local)

Usage:
    from memvid_sdk import create
    from memvid_sdk.clip import get_clip_provider, LocalClip

    # Local CLIP (default, no API key needed)
    clip = get_clip_provider("local")

    # Or with cloud provider
    clip = get_clip_provider("openai")

    # Create memory and store images with CLIP embeddings
    mem = create("gallery.mv2", enable_vec=True)

    embedding = clip.embed_image("photo.jpg")
    mem.put(title="Beach Photo", label="photos", file="photo.jpg", clip_embedding=embedding)

    # Search by text description
    query_embedding = clip.embed_text("sunset over ocean")
    results = mem.find("sunset", query_embedding=query_embedding, mode="clip")
"""

from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Sequence, Union


class ClipProvider(ABC):
    """Abstract base class for CLIP embedding providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name (e.g., 'local:mobileclip-s2')."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension for this model."""
        pass

    @abstractmethod
    def embed_image(self, image_path: str) -> List[float]:
        """Generate embedding for a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text (for text-to-image search).

        Args:
            text: Text description to embed

        Returns:
            Embedding vector
        """
        pass

    def embed_images(self, image_paths: Sequence[str]) -> List[List[float]]:
        """Generate embeddings for multiple images.

        Default implementation calls embed_image for each. Override for batch optimization.

        Args:
            image_paths: List of paths to image files

        Returns:
            List of embedding vectors
        """
        return [self.embed_image(p) for p in image_paths]


class LocalClip(ClipProvider):
    """Local MobileCLIP provider using ONNX runtime.

    Uses MobileCLIP-S2 for fast, offline visual embeddings.
    Model files are loaded from `MEMVID_MODELS_DIR` (default: `~/.memvid/models`).
    Install them once via the CLI:
        - `memvid models install --clip mobileclip-s2`

    Supported models:
        - mobileclip-s2 (default): 512 dims, 101 MB (int8)
        - mobileclip-s2-fp16: 512 dims, 199 MB (better quality)
        - siglip-base: 768 dims, 211 MB (higher accuracy)

    Example:
        >>> clip = LocalClip()
        >>> embedding = clip.embed_image("photo.jpg")
        >>> text_embedding = clip.embed_text("a photo of a cat")
    """

    MODEL_DIMENSIONS = {
        "mobileclip-s2": 512,
        "mobileclip-s2-fp16": 512,
        "siglip-base": 768,
    }

    def __init__(self, model: str = "mobileclip-s2"):
        """Initialize LocalClip provider.

        Args:
            model: Model to use. One of: mobileclip-s2, mobileclip-s2-fp16, siglip-base
        """
        self._model_name = model
        self._model = None
        self._dimension = self.MODEL_DIMENSIONS.get(model, 512)

    @property
    def name(self) -> str:
        return f"local:{self._model_name}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_model(self):
        """Lazy load the CLIP model from native bindings."""
        if self._model is None:
            try:
                from memvid_sdk._lib import ClipModel
                self._model = ClipModel()
            except ImportError as e:
                raise ImportError(
                    "Local CLIP support requires memvid-sdk-free built with CLIP support. "
                    "Install the model files once with `memvid models install --clip mobileclip-s2` "
                    "(or set MEMVID_MODELS_DIR). "
                    f"Original error: {e}"
                )
        return self._model

    def embed_image(self, image_path: str) -> List[float]:
        """Generate embedding for an image using local CLIP model."""
        model = self._get_model()
        return list(model.embed_image(str(image_path)))

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using local CLIP text encoder."""
        model = self._get_model()
        return list(model.embed_text(text))

    def embed_images(self, image_paths: Sequence[str]) -> List[List[float]]:
        """Batch embed multiple images (optimized)."""
        model = self._get_model()
        paths = [str(p) for p in image_paths]
        return [list(e) for e in model.embed_images(paths)]


class OpenAIClip(ClipProvider):
    """OpenAI CLIP-style provider using GPT-4V and text embeddings.

    Workflow:
        1. Image -> GPT-4V generates description -> Description text
        2. Description -> text-embedding-3 -> Embedding vector

    This creates semantically rich, searchable embeddings from images.

    Example:
        >>> clip = OpenAIClip()  # Uses OPENAI_API_KEY env var
        >>> embedding = clip.embed_image("product.jpg")
        >>> query_embedding = clip.embed_text("modern laptop workspace")
    """

    EMBEDDING_MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        vision_model: str = "gpt-4o-mini",
    ):
        """Initialize OpenAI CLIP provider.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
            embedding_model: Model for text embeddings. Default: text-embedding-3-small
            vision_model: Model for image description. Default: gpt-4o-mini
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Pass api_key or set OPENAI_API_KEY environment variable."
            )
        self._embedding_model = embedding_model
        self._vision_model = vision_model
        self._client = None

    @property
    def name(self) -> str:
        return f"openai:{self._embedding_model}"

    @property
    def dimension(self) -> int:
        return self.EMBEDDING_MODEL_DIMENSIONS.get(self._embedding_model, 1536)

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package required for OpenAI CLIP. "
                    "Install with: pip install openai"
                )
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def _describe_image(self, image_path: str) -> str:
        """Use GPT-4V to describe an image."""
        client = self._get_client()

        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        # Determine MIME type
        ext = Path(image_path).suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(ext, "image/jpeg")

        response = client.chat.completions.create(
            model=self._vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in 2-3 detailed sentences for visual search indexing. "
                                "Focus on objects, colors, composition, and scene.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                    },
                ],
            }],
            max_tokens=150,
        )

        return response.choices[0].message.content or ""

    def embed_image(self, image_path: str) -> List[float]:
        """Embed an image by describing it with GPT-4V, then embedding the description."""
        description = self._describe_image(image_path)
        return self.embed_text(description)

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        client = self._get_client()
        response = client.embeddings.create(
            model=self._embedding_model,
            input=text,
        )
        return response.data[0].embedding


class GeminiClip(ClipProvider):
    """Google Gemini multimodal provider.

    Uses Gemini's multimodal capabilities for image understanding
    combined with text embedding for search.

    Example:
        >>> clip = GeminiClip()  # Uses GEMINI_API_KEY env var
        >>> embedding = clip.embed_image("chart.png")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
    ):
        """Initialize Gemini CLIP provider.

        Args:
            api_key: Google AI API key. If not provided, uses GEMINI_API_KEY env var.
            model: Gemini model to use. Default: gemini-2.0-flash
        """
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Gemini API key required. Pass api_key or set GEMINI_API_KEY environment variable."
            )
        self._model = model
        self._client = None
        self._dimension = 768  # Gemini embedding dimension

    @property
    def name(self) -> str:
        return f"gemini:{self._model}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "google-generativeai package required for Gemini CLIP. "
                    "Install with: pip install google-generativeai"
                )
            genai.configure(api_key=self._api_key)
            self._client = genai
        return self._client

    def _describe_image(self, image_path: str) -> str:
        """Use Gemini to describe an image."""
        genai = self._get_client()
        from PIL import Image

        model = genai.GenerativeModel(self._model)
        image = Image.open(image_path)

        response = model.generate_content([
            "Describe this image in 2-3 detailed sentences for visual search indexing. "
            "Focus on objects, colors, composition, and scene.",
            image,
        ])

        return response.text

    def embed_image(self, image_path: str) -> List[float]:
        """Embed an image by describing it with Gemini, then embedding."""
        description = self._describe_image(image_path)
        return self.embed_text(description)

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using Gemini."""
        genai = self._get_client()
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document",
        )
        return result["embedding"]


class HuggingFaceClip(ClipProvider):
    """HuggingFace CLIP provider using transformers.

    Runs CLIP models locally via HuggingFace transformers.
    Supports various CLIP variants including OpenAI CLIP and SigLIP.

    Example:
        >>> clip = HuggingFaceClip()  # Uses openai/clip-vit-base-patch32
        >>> clip = HuggingFaceClip(model="openai/clip-vit-large-patch14")
        >>> embedding = clip.embed_image("photo.jpg")
    """

    def __init__(self, model: str = "openai/clip-vit-base-patch32"):
        """Initialize HuggingFace CLIP provider.

        Args:
            model: HuggingFace model name. Options include:
                   - openai/clip-vit-base-patch32 (default)
                   - openai/clip-vit-large-patch14
                   - google/siglip-base-patch16-224
        """
        self._model_name = model
        self._model = None
        self._processor = None
        self._dimension = 512  # Updated after model load

    @property
    def name(self) -> str:
        return f"huggingface:{self._model_name}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _load_model(self):
        """Lazy load the model and processor."""
        if self._model is None:
            try:
                from transformers import CLIPModel, CLIPProcessor
            except ImportError:
                raise ImportError(
                    "transformers package required for HuggingFace CLIP. "
                    "Install with: pip install transformers torch pillow"
                )
            self._model = CLIPModel.from_pretrained(self._model_name)
            self._processor = CLIPProcessor.from_pretrained(self._model_name)
            self._dimension = self._model.config.projection_dim
        return self._model, self._processor

    def embed_image(self, image_path: str) -> List[float]:
        """Generate embedding for an image."""
        from PIL import Image
        model, processor = self._load_model()
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        outputs = model.get_image_features(**inputs)
        return outputs[0].detach().numpy().tolist()

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        model, processor = self._load_model()
        inputs = processor(text=text, return_tensors="pt", padding=True)
        outputs = model.get_text_features(**inputs)
        return outputs[0].detach().numpy().tolist()


class ReplicateClip(ClipProvider):
    """Replicate API provider for CLIP models.

    Runs CLIP inference via Replicate's hosted models.

    Example:
        >>> clip = ReplicateClip()  # Uses REPLICATE_API_TOKEN env var
        >>> embedding = clip.embed_image("photo.jpg")
    """

    def __init__(
        self,
        api_token: Optional[str] = None,
        model: str = "andreasjansson/clip-features:75b33f253f7714a281ad3e9b28f63e3232d583716ef6718f2e46641077ea040a",
    ):
        """Initialize Replicate CLIP provider.

        Args:
            api_token: Replicate API token. If not provided, uses REPLICATE_API_TOKEN env var.
            model: Replicate model version to use.
        """
        self._api_token = api_token or os.environ.get("REPLICATE_API_TOKEN")
        if not self._api_token:
            raise ValueError(
                "Replicate API token required. Pass api_token or set REPLICATE_API_TOKEN environment variable."
            )
        self._model = model
        self._client = None
        self._dimension = 512

    @property
    def name(self) -> str:
        return f"replicate:clip"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_client(self):
        """Lazy initialization of Replicate client."""
        if self._client is None:
            try:
                import replicate
            except ImportError:
                raise ImportError(
                    "replicate package required. Install with: pip install replicate"
                )
            self._client = replicate.Client(api_token=self._api_token)
        return self._client

    def embed_image(self, image_path: str) -> List[float]:
        """Generate embedding for an image via Replicate."""
        client = self._get_client()
        with open(image_path, "rb") as f:
            output = client.run(
                self._model,
                input={"image": f},
            )
        return list(output)

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text via Replicate."""
        client = self._get_client()
        output = client.run(
            self._model,
            input={"text": text},
        )
        return list(output)


def _parse_provider_model(spec: str) -> tuple[str, Optional[str]]:
    """Parse provider:model specification.

    Examples:
        "openai" -> ("openai", None)
        "openai:gpt-4o" -> ("openai", "gpt-4o")
        "local:siglip-base" -> ("local", "siglip-base")
        "huggingface:openai/clip-vit-large-patch14" -> ("huggingface", "openai/clip-vit-large-patch14")
    """
    if ":" in spec:
        parts = spec.split(":", 1)
        return parts[0].lower(), parts[1]
    return spec.lower(), None


def get_clip_provider(
    provider: str = "local",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> ClipProvider:
    """Factory function to create a CLIP provider.

    Supports two formats:
        1. Separate provider and model: get_clip_provider("openai", model="gpt-4o")
        2. Combined format: get_clip_provider("openai:gpt-4o")

    Args:
        provider: Provider name or "provider:model" format.
                  Supported: local, openai, gemini, huggingface, replicate
        model: Model name (uses provider default if not specified)
        api_key: API key for cloud providers
        **kwargs: Additional provider-specific arguments

    Returns:
        ClipProvider instance

    Examples:
        >>> clip = get_clip_provider("local")
        >>> clip = get_clip_provider("local:siglip-base")
        >>> clip = get_clip_provider("openai")
        >>> clip = get_clip_provider("openai:text-embedding-3-large")
        >>> clip = get_clip_provider("gemini:gemini-2.0-flash")
        >>> clip = get_clip_provider("huggingface:openai/clip-vit-large-patch14")
    """
    # Parse provider:model format
    parsed_provider, parsed_model = _parse_provider_model(provider)

    # Use parsed model if model not explicitly provided
    if model is None:
        model = parsed_model

    provider = parsed_provider

    if provider == "local":
        return LocalClip(model=model or "mobileclip-s2")
    elif provider == "openai":
        return OpenAIClip(
            api_key=api_key,
            embedding_model=model or "text-embedding-3-small",
            **kwargs,
        )
    elif provider == "gemini":
        return GeminiClip(
            api_key=api_key,
            model=model or "gemini-2.0-flash",
            **kwargs,
        )
    elif provider in ("huggingface", "hf"):
        return HuggingFaceClip(model=model or "openai/clip-vit-base-patch32")
    elif provider == "replicate":
        return ReplicateClip(api_token=api_key, **kwargs)
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: local, openai, gemini, huggingface, replicate"
        )


__all__ = [
    "ClipProvider",
    "LocalClip",
    "OpenAIClip",
    "GeminiClip",
    "HuggingFaceClip",
    "ReplicateClip",
    "get_clip_provider",
]
