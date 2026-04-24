"""Python SDK entry point exposing the unified ``use`` factory."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Union
from typing import Literal

from . import _lib

# Base error class - all SDK errors inherit from this
MemvidError = _lib.MemvidError

# MV001: Storage capacity exceeded
CapacityExceededError = _lib.CapacityExceededError

# MV002: Invalid ticket signature
TicketInvalidError = _lib.TicketInvalidError

# MV003: Ticket sequence replay attack
TicketReplayError = _lib.TicketReplayError

# MV004: Lexical index not enabled
LexIndexDisabledError = _lib.LexIndexDisabledError

# MV005: Time index missing or invalid
TimeIndexMissingError = _lib.TimeIndexMissingError

# MV006: File verification failed
VerifyFailedError = _lib.VerifyFailedError

# MV007: File locked by another process
_LockedError = getattr(_lib, "LockedError", None)
if _LockedError is None:
    class LockedError(MemvidError):  # type: ignore[misc]
        """File is locked by another process (MV007)."""
        pass
else:
    LockedError = _LockedError

# MV008: API key required for this operation
_ApiKeyRequiredError = getattr(_lib, "ApiKeyRequiredError", None)
if _ApiKeyRequiredError is None:
    class ApiKeyRequiredError(MemvidError):  # type: ignore[misc]
        """API key required for this operation (MV008)."""
        pass
else:
    ApiKeyRequiredError = _ApiKeyRequiredError

# MV009: Memory already bound to another file
_MemoryAlreadyBoundError = getattr(_lib, "MemoryAlreadyBoundError", None)
if _MemoryAlreadyBoundError is None:
    class MemoryAlreadyBoundError(MemvidError):  # type: ignore[misc]
        """Memory already bound to another file (MV009)."""
        pass
else:
    MemoryAlreadyBoundError = _MemoryAlreadyBoundError

# MV010: Frame not found
_FrameNotFoundError = getattr(_lib, "FrameNotFoundError", None)
if _FrameNotFoundError is None:
    class FrameNotFoundError(MemvidError):  # type: ignore[misc]
        """Requested frame does not exist (MV010)."""
        pass
else:
    FrameNotFoundError = _FrameNotFoundError

# MV011: Vector index not enabled
_VecIndexDisabledError = getattr(_lib, "VecIndexDisabledError", None)
if _VecIndexDisabledError is None:
    class VecIndexDisabledError(MemvidError):  # type: ignore[misc]
        """Vector index not enabled (MV011)."""
        pass
else:
    VecIndexDisabledError = _VecIndexDisabledError

# MV012: Corrupt file detected
_CorruptFileError = getattr(_lib, "CorruptFileError", None)
if _CorruptFileError is None:
    class CorruptFileError(MemvidError):  # type: ignore[misc]
        """File corruption detected (MV012)."""
        pass
else:
    CorruptFileError = _CorruptFileError

# MV013: File not found
_FileNotFoundError = getattr(_lib, "FileNotFoundError", None)
if _FileNotFoundError is None:
    class FileNotFoundError(MemvidError):  # type: ignore[misc]
        """File not found (MV013)."""
        pass
else:
    FileNotFoundError = _FileNotFoundError

# MV014: Vector dimension mismatch
_VecDimensionMismatchError = getattr(_lib, "VecDimensionMismatchError", None)
if _VecDimensionMismatchError is None:
    class VecDimensionMismatchError(MemvidError):  # type: ignore[misc]
        """Vector dimension mismatch (MV014)."""
        pass
else:
    VecDimensionMismatchError = _VecDimensionMismatchError

# MV015: Embedding failed
_EmbeddingFailedError = getattr(_lib, "EmbeddingFailedError", None)
if _EmbeddingFailedError is None:
    class EmbeddingFailedError(MemvidError):  # type: ignore[misc]
        """Embedding failed (MV015)."""
        pass
else:
    EmbeddingFailedError = _EmbeddingFailedError

# MV016: Encryption/decryption error (.mv2e)
_EncryptionError = getattr(_lib, "EncryptionError", None)
if _EncryptionError is None:
    class EncryptionError(MemvidError):  # type: ignore[misc]
        """Encryption/decryption error (.mv2e) (MV016)."""
        pass
else:
    EncryptionError = _EncryptionError

# MV017: NER model not available
_NerModelNotAvailableError = getattr(_lib, "NerModelNotAvailableError", None)
if _NerModelNotAvailableError is None:
    class NerModelNotAvailableError(MemvidError):  # type: ignore[misc]
        """NER model not available (MV017)."""
        pass
else:
    NerModelNotAvailableError = _NerModelNotAvailableError

# MV018: CLIP index not enabled
_ClipIndexDisabledError = getattr(_lib, "ClipIndexDisabledError", None)
if _ClipIndexDisabledError is None:
    class ClipIndexDisabledError(MemvidError):  # type: ignore[misc]
        """CLIP index not enabled (MV018)."""
        pass
else:
    ClipIndexDisabledError = _ClipIndexDisabledError

# MV023: Query quota exceeded (HTTP 429)
_QuotaExceededError = getattr(_lib, "QuotaExceededError", None)
if _QuotaExceededError is None:
    class QuotaExceededError(MemvidError):  # type: ignore[misc]
        """Monthly query quota exceeded (MV023).

        Raised when the organisation's query limit for the billing period
        has been exceeded. Upgrade your plan or wait for the quota to reset.

        Attributes:
            limit: The monthly query limit for the plan.
            used: Number of queries used in the current period.
            reset_date: ISO 8601 timestamp when the quota resets.
            plan_name: The current plan name.
        """
        def __init__(self, message: str, limit: int = 0, used: int = 0,
                     reset_date: str = "", plan_name: str = ""):
            super().__init__(message)
            self.limit = limit
            self.used = used
            self.reset_date = reset_date
            self.plan_name = plan_name
else:
    QuotaExceededError = _QuotaExceededError


# Error code suggestions - helps users resolve common issues
_ERROR_SUGGESTIONS: Dict[str, str] = {
    "MV001": "Upgrade your plan at https://memvid.com/dashboard/plan or create a new memory.",
    "MV002": "Your ticket may have expired. Re-sync with: memvid config check",
    "MV003": "Ticket replay detected. This usually means the memory was restored from backup. Contact support if this persists.",
    "MV004": "Enable lexical search with: memvid doctor --rebuild-lex-index <file>",
    "MV005": "Rebuild the time index with: memvid doctor --rebuild-time-index <file>",
    "MV006": "The file may be corrupted. Try: memvid doctor <file> or restore from backup.",
    "MV007": "The file is locked by another process. Release it with: memvid lock nudge <file>",
    "MV008": "Set your API key with: memvid config set api-key <your-key>\nGet a key at https://memvid.com/dashboard/api-keys",
    "MV009": "This local file is already bound to a cloud memory. Use unbind() first or create a new local file.",
    "MV010": "The requested frame does not exist. Check the frame ID with: memvid timeline <file>",
    "MV011": "Enable vector search with: memvid doctor --rebuild-vec-index <file>",
    "MV012": "The file appears corrupted. Run: memvid verify <file> for details. Restore from backup if needed.",
    "MV013": "Check the file path and ensure it exists. Create a new memory with: memvid create <file>",
    "MV014": "Ensure all embeddings use the same model. Check memory stats with: memvid stats <file>",
    "MV015": "Check your embedding provider API key and network connection.",
    "MV016": "Check your password. If forgotten, the encrypted file cannot be recovered.",
    "MV017": "Install the NER model or use a different extraction engine.",
    "MV018": "Enable CLIP index for image search capabilities.",
    "MV020": "Check your API key and network connection. Validate with: memvid config check",
    "MV021": "Check your API key and network connection. Validate with: memvid config check",
    "MV022": "Check your API key and network connection. Validate with: memvid config check",
    "MV023": "Your monthly query quota has been exceeded. Upgrade your plan at https://memvid.com/dashboard/plan",
}


def get_error_suggestion(code: str) -> Optional[str]:
    """Get a user-friendly suggestion for how to resolve an error.

    Args:
        code: The error code (e.g., "MV001", "MV007").

    Returns:
        A suggestion string, or None if no suggestion is available.

    Example:
        >>> try:
        ...     mv = create("locked.mv2")
        ... except LockedError as e:
        ...     print(f"Error: {e}")
        ...     print(f"Suggestion: {get_error_suggestion('MV007')}")
    """
    return _ERROR_SUGGESTIONS.get(code)


def format_error_with_suggestion(error: MemvidError) -> str:
    """Format an error message with its suggestion.

    Args:
        error: A MemvidError instance.

    Returns:
        A formatted string with the error and suggestion.

    Example:
        >>> try:
        ...     mv = create("locked.mv2")
        ... except MemvidError as e:
        ...     print(format_error_with_suggestion(e))
    """
    # Extract code from error message if possible
    msg = str(error)
    code = None
    if msg.startswith("MV") and len(msg) > 5 and msg[5] == ":":
        code = msg[:5]

    suggestion = get_error_suggestion(code) if code else None
    if suggestion:
        return f"{msg}\n\nSuggestion: {suggestion}"
    return msg


_MemvidCore = _lib._MemvidCore
_open = _lib.open
_put = _lib.put
_find = _lib.find
_ask = _lib.ask
_verify = getattr(_lib, "verify", None)
_lock_who = getattr(_lib, "lock_who", None)
_lock_nudge = getattr(_lib, "lock_nudge", None)
_lock_capsule = getattr(_lib, "lock", None)
_unlock_capsule = getattr(_lib, "unlock", None)
_version_info = getattr(_lib, "version_info", None)
from ._registry import registry
from ._sentinel import NoOp
from ._analytics import track_command, flush as flush_analytics, is_telemetry_enabled

# Ensure adapter modules register their loaders.
from . import adapters as _adapters  # noqa: F401

# Import embeddings module
from . import embeddings

# Import CLIP and entities modules
from . import clip
from . import entities

# Stable kind identifiers shared across bindings.
Kind = Literal[
    "basic",
    "langchain",
    "llamaindex",
    "crewai",
    "vercel-ai",
    "openai",
    "autogen",
    "haystack",
    "langgraph",
    "semantic-kernel",
    "mcp",
]

ApiKey = Union[str, Mapping[str, str]]

_MEMVID_EMBEDDING_PROVIDER_KEY = "memvid.embedding.provider"
_MEMVID_EMBEDDING_MODEL_KEY = "memvid.embedding.model"
_MEMVID_EMBEDDING_DIMENSION_KEY = "memvid.embedding.dimension"
_MEMVID_EMBEDDING_NORMALIZED_KEY = "memvid.embedding.normalized"


def _normalise_apikey(apikey: Optional[ApiKey]) -> Optional[Dict[str, str]]:
    if apikey is None:
        return None
    if isinstance(apikey, str):
        return {"default": apikey}
    return {str(key): str(value) for key, value in apikey.items()}


def _apply_embedding_identity_metadata(
    metadata: MutableMapping[str, Any],
    identity: Mapping[str, Any],
    embedding_dimension: Optional[int],
) -> None:
    provider_raw = identity.get("provider")
    model_raw = identity.get("model")
    dimension_raw = identity.get("dimension", embedding_dimension)
    normalized_raw = identity.get("normalized")

    provider = str(provider_raw).strip().lower() if provider_raw is not None else None
    model = str(model_raw).strip() if model_raw is not None else None

    if not provider and not model:
        return

    if provider:
        metadata[_MEMVID_EMBEDDING_PROVIDER_KEY] = provider
    if model:
        metadata[_MEMVID_EMBEDDING_MODEL_KEY] = model

    if dimension_raw is not None:
        try:
            dimension = int(dimension_raw)
        except (TypeError, ValueError):
            dimension = None
        if dimension and dimension > 0:
            metadata[_MEMVID_EMBEDDING_DIMENSION_KEY] = dimension

    if normalized_raw is not None:
        metadata[_MEMVID_EMBEDDING_NORMALIZED_KEY] = bool(normalized_raw)


def _embedding_identity_from_embedder(embedder: "embeddings.EmbeddingProvider") -> Dict[str, Any]:
    provider = embedder.__class__.__name__.lower()
    if isinstance(embedder, embeddings.OpenAIEmbeddings):
        provider = "openai"
    elif isinstance(embedder, embeddings.CohereEmbeddings):
        provider = "cohere"
    elif isinstance(embedder, embeddings.VoyageEmbeddings):
        provider = "voyage"
    elif hasattr(embeddings, "NvidiaEmbeddings") and isinstance(embedder, embeddings.NvidiaEmbeddings):
        provider = "nvidia"
    elif isinstance(embedder, embeddings.HuggingFaceEmbeddings):
        provider = "huggingface"
    elif hasattr(embeddings, "HashEmbeddings") and isinstance(embedder, getattr(embeddings, "HashEmbeddings")):
        provider = "custom"

    return {
        "provider": provider,
        "model": getattr(embedder, "model_name", None),
        "dimension": getattr(embedder, "dimension", None),
        "normalized": None,
    }


# ===========================================================================
# Global Configuration
# ===========================================================================

from dataclasses import dataclass, field
from typing import TypedDict


class MemvidConfig(TypedDict, total=False):
    """Global configuration for the Memvid SDK.

    Set once at startup, then used as defaults for all operations.
    """
    api_key: str
    """Memvid API key (mv2_*) for dashboard operations."""
    dashboard_url: str
    """Dashboard URL (default: https://memvid.com)."""
    default_memory: str
    """Default memory ID to use when none is specified."""
    memories: Dict[str, str]
    """Named memory aliases (e.g., {"work": "abc123", "personal": "def456"})."""
    default_embedding_provider: str
    """Default embedding provider (openai, cohere, voyage, etc.)."""
    default_llm_provider: str
    """Default LLM provider for ask() operations."""


_global_config: MemvidConfig = {}


class ApiKeyAclScope(TypedDict, total=False):
    """Normalized ACL scope attached to an API key."""
    tenant_id: Optional[str]
    """Tenant identifier (required for enforce mode)."""
    subject_id: Optional[str]
    """Optional principal/subject identifier."""
    roles: List[str]
    """Normalized roles."""
    group_ids: List[str]
    """Normalized group identifiers."""
    enforcement_mode: Literal["audit", "enforce"]
    """ACL enforcement mode."""


_DEFAULT_ACL_SCOPE: ApiKeyAclScope = {
    "tenant_id": None,
    "subject_id": None,
    "roles": [],
    "group_ids": [],
    "enforcement_mode": "audit",
}
_acl_scope_cache: Dict[str, ApiKeyAclScope] = {}


def _resolve_dashboard_url(override: Optional[str] = None) -> str:
    return (
        override
        or _global_config.get("dashboard_url")
        or os.environ.get("MEMVID_DASHBOARD_URL")
        or "https://memvid.com"
    ).rstrip("/")


def _normalise_acl_string(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    return trimmed.lower()


def _normalise_acl_list(value: Any) -> List[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    out = {
        normalized
        for normalized in (_normalise_acl_string(item) for item in value)
        if normalized
    }
    return sorted(out)


def _normalise_acl_scope(scope: Optional[Mapping[str, Any]]) -> ApiKeyAclScope:
    if not isinstance(scope, Mapping):
        return dict(_DEFAULT_ACL_SCOPE)

    tenant_id = _normalise_acl_string(scope.get("tenant_id", scope.get("tenantId")))
    subject_id = _normalise_acl_string(scope.get("subject_id", scope.get("subjectId")))
    roles = _normalise_acl_list(scope.get("roles"))
    group_ids = _normalise_acl_list(scope.get("group_ids", scope.get("groupIds")))
    enforcement_mode = (
        "enforce"
        if scope.get("enforcement_mode", scope.get("enforcementMode")) == "enforce"
        else "audit"
    )
    return {
        "tenant_id": tenant_id,
        "subject_id": subject_id,
        "roles": roles,
        "group_ids": group_ids,
        "enforcement_mode": enforcement_mode,
    }


def configure(config: MemvidConfig) -> None:
    """Configure global defaults for the Memvid SDK.

    Priority order for settings:
    1. Explicit function arguments
    2. Global config (set via configure())
    3. Environment variables

    Args:
        config: Configuration dictionary with optional keys for api_key,
                dashboard_url, default_memory, memories, etc.

    Raises:
        ApiKeyRequiredError: If api_key is provided but doesn't start with 'mv2_'.

    Example:
        >>> from memvid_sdk import configure, create
        >>>
        >>> # Set up once at startup
        >>> configure({
        ...     "api_key": "mv2_your_api_key",
        ...     "dashboard_url": "http://localhost:3001",  # for local dev
        ...     "default_memory": "abc123",
        ...     "memories": {
        ...         "work": "abc123",
        ...         "personal": "def456",
        ...     },
        ... })
        >>>
        >>> # Now create() uses defaults automatically
        >>> mem = create("data.mv2")
    """
    global _global_config

    # Validate API key format if provided
    api_key = config.get("api_key")
    if api_key and not api_key.strip().startswith("mv2_"):
        raise ApiKeyRequiredError(
            "Invalid API key format. Expected mv2_* prefix. "
            "Get your API key at https://memvid.com/dashboard/api-keys"
        )

    _global_config = {**_global_config, **config}


def get_config() -> MemvidConfig:
    """Get the current global configuration.

    Returns a copy to prevent external mutation.

    Returns:
        Copy of the current global configuration.
    """
    return dict(_global_config)


def reset_config() -> None:
    """Reset global configuration to defaults.

    Useful for testing or reinitializing.
    """
    global _global_config
    _global_config = {}
    _acl_scope_cache.clear()


def get_acl_scope_from_api_key(
    *,
    api_key: Optional[str] = None,
    dashboard_url: Optional[str] = None,
    timeout_sec: float = 5.0,
    force_refresh: bool = False,
) -> ApiKeyAclScope:
    """Fetch ACL scope attached to a Memvid API key via ``/api/ticket``.

    Args:
        api_key: Memvid API key. Falls back to configured/global/env key.
        dashboard_url: Dashboard base URL. Falls back to configured/global/env URL.
        timeout_sec: HTTP timeout in seconds (default: 5.0).
        force_refresh: Ignore in-memory cache and fetch fresh scope.

    Returns:
        Normalized ACL scope dict with ``tenant_id``, ``subject_id``, ``roles``,
        ``group_ids``, and ``enforcement_mode``.
    """
    import urllib.request
    import urllib.error
    import json

    effective_api_key = (
        api_key
        or _global_config.get("api_key")
        or os.environ.get("MEMVID_API_KEY")
        or ""
    ).strip()
    if not effective_api_key:
        raise ApiKeyRequiredError(
            "API key required for get_acl_scope_from_api_key(). "
            "Set via configure({'api_key': 'mv2_...'}) or MEMVID_API_KEY."
        )
    if not effective_api_key.startswith("mv2_"):
        raise ApiKeyRequiredError(
            "Invalid API key format. Expected mv2_* prefix."
        )

    resolved_dashboard_url = _resolve_dashboard_url(dashboard_url)
    cache_key = f"{resolved_dashboard_url}|{effective_api_key}"
    if not force_refresh and cache_key in _acl_scope_cache:
        return dict(_acl_scope_cache[cache_key])

    req = urllib.request.Request(
        f"{resolved_dashboard_url}/api/ticket",
        method="GET",
        headers={"x-api-key": effective_api_key},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8") if e.fp else ""
        if e.code == 401:
            raise MemvidError(
                "MV022: Invalid API key. Get a valid key at https://memvid.com/dashboard/api-keys"
            )
        raise MemvidError(
            f"MV022: Failed to fetch ACL scope ({e.code} {e.reason}). {body_text}"
        )
    except Exception as e:
        raise MemvidError(f"MV022: Failed to fetch ACL scope: {e}")

    envelope = payload.get("data", payload) if isinstance(payload, Mapping) else {}
    scope = _normalise_acl_scope(
        envelope.get("acl_scope") if isinstance(envelope, Mapping) else None
    )
    _acl_scope_cache[cache_key] = dict(scope)
    return dict(scope)


def acl_context_from_scope(
    scope: Optional[Mapping[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Convert API-key ACL scope into query ``acl_context`` payload.

    Returns ``None`` when no tenant scope is present.
    """
    normalized = _normalise_acl_scope(scope)
    tenant_id = normalized.get("tenant_id")
    if not tenant_id:
        return None
    context: Dict[str, Any] = {"tenant_id": tenant_id}
    subject_id = normalized.get("subject_id")
    roles = normalized.get("roles") or []
    group_ids = normalized.get("group_ids") or []
    if subject_id:
        context["subject_id"] = subject_id
    if roles:
        context["roles"] = list(roles)
    if group_ids:
        context["group_ids"] = list(group_ids)
    return context


def acl_metadata_from_scope(
    scope: Optional[Mapping[str, Any]],
    *,
    visibility: Literal["auto", "public", "restricted"] = "auto",
    policy_version: str = "v1",
) -> Dict[str, Any]:
    """Build ingest metadata ACL fields from normalized key scope."""
    normalized = _normalise_acl_scope(scope)
    tenant_id = normalized.get("tenant_id")
    if not tenant_id:
        raise MemvidError(
            "MV999: ACL scope is missing tenant_id. Create an ACL-scoped API key before ingesting restricted data."
        )
    if visibility not in ("auto", "public", "restricted"):
        raise ValueError("visibility must be one of: auto, public, restricted")

    subject_id = normalized.get("subject_id")
    roles = normalized.get("roles") or []
    group_ids = normalized.get("group_ids") or []
    has_readers = bool(subject_id or roles or group_ids)
    effective_visibility = "restricted" if visibility == "auto" and has_readers else (
        "public" if visibility == "auto" else visibility
    )

    metadata: Dict[str, Any] = {
        "acl_tenant_id": tenant_id,
        "acl_visibility": effective_visibility,
        "acl_policy_version": policy_version,
    }
    if subject_id:
        metadata["acl_read_principals"] = [subject_id]
    if roles:
        metadata["acl_read_roles"] = list(roles)
    if group_ids:
        metadata["acl_read_groups"] = list(group_ids)
    return metadata


class ConfigValidationResult(TypedDict, total=False):
    """Result of validating a single configuration item."""
    configured: bool
    """Whether the configuration is set."""
    valid: bool
    """Whether the configuration is valid (tested against API)."""
    error: str
    """Error message if validation failed."""


class ValidateConfigResult(TypedDict, total=False):
    """Result of validating all configuration items."""
    memvid: ConfigValidationResult
    """Memvid API key validation result."""
    dashboard: ConfigValidationResult
    """Dashboard URL reachability result."""
    groq: ConfigValidationResult
    """Groq API key validation result (optional)."""
    openai: ConfigValidationResult
    """OpenAI API key validation result (optional)."""
    gemini: ConfigValidationResult
    """Gemini API key validation result (optional)."""
    anthropic: ConfigValidationResult
    """Anthropic API key validation result (optional)."""
    all_valid: bool
    """Summary: True if all configured items are valid."""


def validate_config(*, check_llm_providers: bool = False) -> ValidateConfigResult:
    """Validate the current configuration by testing API connectivity.

    Args:
        check_llm_providers: Whether to validate LLM provider keys (default: False).

    Returns:
        Validation results for all configured items.

    Example:
        >>> from memvid_sdk import configure, validate_config
        >>> configure({
        ...     "api_key": "mv2_your_api_key",
        ...     "dashboard_url": "http://localhost:3001",
        ... })
        >>> result = validate_config()
        >>> if result.get("all_valid"):
        ...     print("All configured!")
        >>> # Check LLM providers too
        >>> full_result = validate_config(check_llm_providers=True)
    """
    import urllib.request
    import urllib.error
    import json

    config = get_config()

    # Default dashboard URL
    dashboard_url = (
        config.get("dashboard_url")
        or os.environ.get("MEMVID_DASHBOARD_URL")
        or "https://memvid.com"
    ).rstrip("/")
    api_key = config.get("api_key") or os.environ.get("MEMVID_API_KEY")

    result: ValidateConfigResult = {
        "memvid": {"configured": False, "valid": False},
        "dashboard": {"configured": True, "valid": False},
        "all_valid": False,
    }

    # Check dashboard reachability
    try:
        health_url = f"{dashboard_url}/api/health"
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            result["dashboard"]["valid"] = resp.status == 200
    except urllib.error.HTTPError as e:
        result["dashboard"]["error"] = f"Dashboard returned {e.code}"
    except Exception as e:
        result["dashboard"]["error"] = str(e)

    # Check Memvid API key
    if api_key:
        result["memvid"]["configured"] = True
        try:
            ticket_url = f"{dashboard_url}/api/ticket"
            req = urllib.request.Request(ticket_url, method="GET")
            req.add_header("x-api-key", api_key)
            with urllib.request.urlopen(req, timeout=5) as resp:
                result["memvid"]["valid"] = resp.status == 200
        except urllib.error.HTTPError as e:
            result["memvid"]["error"] = "Invalid API key" if e.code == 401 else f"API returned {e.code}"
        except Exception as e:
            result["memvid"]["error"] = str(e)

    # Check LLM providers (optional)
    if check_llm_providers:
        # Groq
        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key:
            result["groq"] = {"configured": True, "valid": False}
            try:
                req = urllib.request.Request("https://api.groq.com/openai/v1/models", method="GET")
                req.add_header("Authorization", f"Bearer {groq_key}")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    result["groq"]["valid"] = resp.status == 200
            except urllib.error.HTTPError as e:
                result["groq"]["error"] = f"Groq API returned {e.code}"
            except Exception as e:
                result["groq"]["error"] = str(e)
        else:
            result["groq"] = {"configured": False, "valid": False}

        # OpenAI
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            result["openai"] = {"configured": True, "valid": False}
            try:
                req = urllib.request.Request("https://api.openai.com/v1/models", method="GET")
                req.add_header("Authorization", f"Bearer {openai_key}")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    result["openai"]["valid"] = resp.status == 200
            except urllib.error.HTTPError as e:
                result["openai"]["error"] = f"OpenAI API returned {e.code}"
            except Exception as e:
                result["openai"]["error"] = str(e)
        else:
            result["openai"] = {"configured": False, "valid": False}

        # Gemini
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            result["gemini"] = {"configured": True, "valid": False}
            try:
                req = urllib.request.Request(
                    f"https://generativelanguage.googleapis.com/v1/models?key={gemini_key}",
                    method="GET",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    result["gemini"]["valid"] = resp.status == 200
            except urllib.error.HTTPError as e:
                result["gemini"]["error"] = f"Gemini API returned {e.code}"
            except Exception as e:
                result["gemini"]["error"] = str(e)
        else:
            result["gemini"] = {"configured": False, "valid": False}

        # Anthropic
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            result["anthropic"] = {"configured": True, "valid": False}
            # Anthropic doesn't have a simple models endpoint, check key format
            result["anthropic"]["valid"] = anthropic_key.startswith("sk-ant-")
            if not result["anthropic"]["valid"]:
                result["anthropic"]["error"] = "Invalid key format (expected sk-ant-*)"
        else:
            result["anthropic"] = {"configured": False, "valid": False}

    # Compute all_valid
    memvid_ok = not result["memvid"].get("configured") or result["memvid"].get("valid")
    groq_ok = not result.get("groq", {}).get("configured") or result.get("groq", {}).get("valid")
    openai_ok = not result.get("openai", {}).get("configured") or result.get("openai", {}).get("valid")
    gemini_ok = not result.get("gemini", {}).get("configured") or result.get("gemini", {}).get("valid")
    anthropic_ok = not result.get("anthropic", {}).get("configured") or result.get("anthropic", {}).get("valid")

    result["all_valid"] = (
        result["dashboard"].get("valid", False)
        and memvid_ok
        and groq_ok
        and openai_ok
        and gemini_ok
        and anthropic_ok
    )

    return result


def resolve_memory(name_or_id: str) -> str:
    """Resolve a memory name to its ID.

    Checks named memories first, then returns the input if not found.

    Args:
        name_or_id: Memory name (from configure()) or ID.

    Returns:
        The resolved memory ID.

    Example:
        >>> configure({"memories": {"work": "abc123"}})
        >>> resolve_memory("work")  # Returns 'abc123'
        >>> resolve_memory("xyz789")  # Returns 'xyz789' (passthrough)
    """
    memories = _global_config.get("memories", {})
    return memories.get(name_or_id, name_or_id)


# ===========================================================================
# Query Usage Tracking
# ===========================================================================


def _track_query_usage(api_key: str, count: int = 1) -> None:
    """Track a query against the user's plan quota.

    Called before find() and ask() operations when API key is configured.

    Args:
        api_key: The Memvid API key (mv2_xxx)
        count: Number of queries to track (default: 1)

    Raises:
        QuotaExceededError: If the monthly limit is reached
    """
    import urllib.request
    import urllib.error
    import json

    dashboard_url = (
        _global_config.get("dashboard_url")
        or os.environ.get("MEMVID_DASHBOARD_URL")
        or "https://memvid.com"
    ).rstrip("/")

    url = f"{dashboard_url}/api/v1/query"

    try:
        data = json.dumps({"count": count}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key,
            },
            method="POST",
        )

        # 5 second timeout
        with urllib.request.urlopen(req, timeout=5) as response:
            pass  # Success - query tracked

    except urllib.error.HTTPError as e:
        if e.code == 429:
            # Quota exceeded
            try:
                response_data = json.loads(e.read().decode("utf-8"))
            except Exception:
                response_data = {}

            raise QuotaExceededError(
                response_data.get("message", "Monthly query quota exceeded"),
                limit=response_data.get("limit"),
                used=response_data.get("used"),
                reset_date=response_data.get("resetDate"),
                plan_name=response_data.get("planName"),
            )
        else:
            # Log but don't fail the query - tracking is best-effort
            import warnings
            warnings.warn(f"[memvid] Query tracking failed: HTTP {e.code}", stacklevel=2)

    except Exception as e:
        # Log but don't fail the query for network errors
        import warnings
        warnings.warn(f"[memvid] Query tracking error: {e}", stacklevel=2)


# ===========================================================================
# Cloud Memory Management
# ===========================================================================


class CreateMemoryResult(TypedDict, total=False):
    """Result of creating a memory on the cloud."""
    id: str
    """The memory ID (24-char ObjectId format)."""
    organisation_id: str
    """Organisation ID."""
    project_id: str
    """Project ID."""
    slug: str
    """URL-friendly slug."""
    display_name: str
    """Display name."""
    description: Optional[str]
    """Description."""
    capacity_bytes: int
    """Capacity in bytes."""
    created_at: str
    """Creation timestamp."""
    updated_at: str
    """Last update timestamp."""


class MemoryInfo(TypedDict, total=False):
    """Memory info returned from list_memories."""
    id: str
    """The memory ID (24-char ObjectId format)."""
    organisation_id: str
    """Organisation ID."""
    project_id: str
    """Project ID."""
    slug: str
    """URL-friendly slug."""
    display_name: str
    """Display name."""
    description: Optional[str]
    """Description."""
    capacity_bytes: int
    """Capacity in bytes."""


class CreateProjectResult(TypedDict, total=False):
    """Result of creating a project on the cloud."""
    id: str
    """The project ID (24-char ObjectId format)."""
    organisation_id: str
    """Organisation ID."""
    slug: str
    """URL-friendly slug."""
    name: str
    """Project name."""
    description: Optional[str]
    """Description."""
    created_at: str
    """Creation timestamp."""
    updated_at: str
    """Last update timestamp."""


class ProjectInfo(TypedDict, total=False):
    """Project info returned from list_projects."""
    id: str
    """The project ID (24-char ObjectId format)."""
    organisation_id: str
    """Organisation ID."""
    slug: str
    """URL-friendly slug."""
    name: str
    """Project name."""
    description: Optional[str]
    """Description."""
    created_at: str
    """Creation timestamp."""
    updated_at: str
    """Last update timestamp."""


# ===========================================================================
# Response Types for SDK Methods
# ===========================================================================


class StatsResult(TypedDict, total=False):
    """Response from stats() method."""
    active_frame_count: int
    """Number of active (non-deleted) frames."""
    average_frame_logical_bytes: int
    """Average logical bytes per frame."""
    average_frame_payload_bytes: int
    """Average payload bytes per frame."""
    capacity_bytes: int
    """Total capacity in bytes."""
    compression_ratio_percent: float
    """Compression ratio as percentage."""
    effective_vec_dimension: Optional[int]
    """Vector dimension if vectors are stored."""
    frame_count: int
    """Total number of frames."""
    has_lex_index: bool
    """Whether lexical index is enabled."""
    has_time_index: bool
    """Whether time index is enabled."""
    has_vec_index: bool
    """Whether vector index is enabled."""
    logical_bytes: int
    """Total logical bytes of content."""
    payload_bytes: int
    """Total payload bytes stored."""
    remaining_capacity_bytes: int
    """Remaining capacity in bytes."""
    saved_bytes: int
    """Bytes saved via compression."""
    savings_percent: float
    """Savings percentage via compression."""
    seq_no: int
    """Current sequence number."""
    size_bytes: int
    """File size in bytes."""
    storage_utilisation_percent: float
    """Storage utilisation as percentage."""
    tier: str
    """Current tier (dev, starter, pro, etc.)."""


class FindHit(TypedDict, total=False):
    """A single hit from find() or vec_search()."""
    frame_id: int
    """Frame ID."""
    uri: str
    """URI of the frame."""
    title: str
    """Title of the frame."""
    snippet: str
    """Snippet of matching content."""
    score: float
    """Relevance score."""
    rank: int
    """Rank in results."""
    matches: int
    """Number of term matches (lexical only)."""
    tags: List[str]
    """Tags associated with the frame."""
    labels: List[str]
    """Labels associated with the frame."""
    track: Optional[str]
    """Track name if applicable."""
    content_dates: List[str]
    """Dates extracted from content."""
    created_at: str
    """Creation timestamp."""


class FindResult(TypedDict, total=False):
    """Response from find() method."""
    query: str
    """The search query."""
    engine: str
    """Search engine used (tantivy, etc.)."""
    hits: List[FindHit]
    """List of matching hits."""
    total_hits: int
    """Total number of matches."""
    context: str
    """Aggregated context string."""
    next_cursor: Optional[str]
    """Cursor for pagination."""
    took_ms: int
    """Time taken in milliseconds."""


class VecSearchResult(TypedDict, total=False):
    """Response from vec_search() method."""
    query: str
    """The search query."""
    hits: List[FindHit]
    """List of matching hits."""
    total_hits: int
    """Total number of matches."""
    context: str
    """Aggregated context string."""
    took_ms: int
    """Time taken in milliseconds."""


class AskStats(TypedDict, total=False):
    """Stats from ask() response."""
    latency_ms: int
    """Total latency in milliseconds."""
    retrieval_ms: int
    """Retrieval time in milliseconds."""
    synthesis_ms: int
    """LLM synthesis time in milliseconds."""


class AskUsage(TypedDict, total=False):
    """Usage info from ask() response."""
    retrieved: int
    """Number of frames retrieved."""
    prompt_tokens: int
    """Prompt tokens used."""
    completion_tokens: int
    """Completion tokens used."""
    total_tokens: int
    """Total tokens used."""


class AskSource(TypedDict, total=False):
    """Source reference from ask() response."""
    frame_id: int
    """Frame ID."""
    uri: str
    """URI of the source frame."""
    title: str
    """Title of the source."""
    snippet: str
    """Snippet from source."""
    score: float
    """Relevance score."""


class FollowUp(TypedDict, total=False):
    """Follow-up suggestions when answer has low confidence."""
    needed: bool
    """Whether follow-up is needed."""
    reason: str
    """Why confidence is low."""
    hint: str
    """Helpful hint for the user."""
    available_topics: List[str]
    """Topics available in this memory."""
    suggestions: List[str]
    """Suggested questions to ask."""


class Grounding(TypedDict, total=False):
    """Grounding information for answer quality assessment."""
    score: float
    """Overall grounding score (0-1)."""
    label: str
    """Grounding quality label (LOW, MEDIUM, HIGH)."""
    sentence_count: int
    """Number of sentences in the answer."""
    grounded_sentences: int
    """Number of sentences grounded in context."""
    has_warning: bool
    """Whether there's a grounding warning."""
    warning_reason: str
    """Reason for the warning if any."""


class AskResult(TypedDict, total=False):
    """Response from ask() method."""
    question: str
    """The question asked."""
    answer: Optional[str]
    """The generated answer (None if context_only=True)."""
    context: str
    """Aggregated context used for answer."""
    context_fragments: List[str]
    """Individual context fragments."""
    context_only: bool
    """Whether this was a context-only request."""
    mode: str
    """Search mode used (lex, sem, auto)."""
    retriever: str
    """Retriever used."""
    hits: List[FindHit]
    """Retrieved hits."""
    sources: List[AskSource]
    """Source references."""
    stats: AskStats
    """Performance stats."""
    usage: AskUsage
    """Token usage."""
    model: str
    """Model used for generation."""
    grounding: Grounding
    """Grounding information for answer quality assessment."""
    follow_up: FollowUp
    """Follow-up suggestions when confidence is low."""


class TimelineEntry(TypedDict, total=False):
    """A single entry from timeline()."""
    frame_id: int
    """Frame ID."""
    uri: str
    """URI of the frame."""
    timestamp: int
    """Unix timestamp."""
    preview: str
    """Preview text."""
    child_frames: List[int]
    """Child frame IDs (for chunked documents)."""


def create_memory(
    name: str,
    *,
    description: Optional[str] = None,
    project_id: Optional[str] = None,
    api_key: Optional[str] = None,
    dashboard_url: Optional[str] = None,
) -> CreateMemoryResult:
    """Create a new memory on the Memvid cloud.

    This creates a memory entry in your dashboard that can be used to:
    - Track storage usage against your plan limits
    - Sync local .mv2 files to the cloud
    - Share memories across devices

    Args:
        name: Display name for the memory.
        description: Optional description.
        project_id: Optional project ID (uses default project if not specified).
        api_key: API key (uses global config or env var if not specified).
        dashboard_url: Dashboard URL (uses global config or env var if not specified).

    Returns:
        The created memory info including the ID.

    Example:
        >>> from memvid_sdk import configure, create_memory, create
        >>> configure({
        ...     "api_key": "mv2_your_api_key",
        ...     "dashboard_url": "http://localhost:3001",
        ... })
        >>> # Create a cloud memory
        >>> memory = create_memory(
        ...     "Customer Support KB",
        ...     description="Knowledge base for support team",
        ... )
        >>> print("Created memory:", memory["id"])
        >>> # Now create a local .mv2 file bound to this memory
        >>> mv = create("support.mv2", memory_id=memory["id"])
        >>> mv.put("docs.pdf")
    """
    import urllib.request
    import urllib.error
    import json

    # Resolve API key
    effective_api_key = (
        api_key
        or _global_config.get("api_key")
        or os.environ.get("MEMVID_API_KEY")
    )
    if not effective_api_key:
        raise ApiKeyRequiredError(
            "API key required for create_memory(). "
            "Set via configure({'api_key': 'mv2_...'}), MEMVID_API_KEY env var, or api_key parameter. "
            "Get your API key at https://memvid.com/dashboard/api-keys"
        )

    # Resolve dashboard URL
    effective_dashboard_url = (
        dashboard_url
        or _global_config.get("dashboard_url")
        or os.environ.get("MEMVID_DASHBOARD_URL")
        or "https://memvid.com"
    ).rstrip("/")

    url = f"{effective_dashboard_url}/api/memories"

    body: Dict[str, Any] = {"name": name}
    if description:
        body["description"] = description
    if project_id:
        body["projectId"] = project_id

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("x-api-key", effective_api_key)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            response_data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8") if e.fp else ""
        raise MemvidError(f"Failed to create memory: {e.code} {e.reason}. {body_text}")
    except Exception as e:
        raise MemvidError(f"Failed to create memory: {e}")

    result_data = response_data.get("data", response_data)

    return CreateMemoryResult(
        id=result_data.get("id"),
        organisation_id=result_data.get("organisation_id"),
        project_id=result_data.get("project_id"),
        slug=result_data.get("slug"),
        display_name=result_data.get("display_name"),
        description=result_data.get("description"),
        capacity_bytes=result_data.get("capacity_bytes"),
        created_at=result_data.get("created_at"),
        updated_at=result_data.get("updated_at"),
    )


def list_memories(
    *,
    project_id: Optional[str] = None,
    api_key: Optional[str] = None,
    dashboard_url: Optional[str] = None,
) -> List[MemoryInfo]:
    """List all memories in your organisation.

    Args:
        project_id: Optional project ID to filter by.
        api_key: API key (uses global config or env var if not specified).
        dashboard_url: Dashboard URL (uses global config or env var if not specified).

    Returns:
        List of memory info dictionaries.

    Example:
        >>> from memvid_sdk import configure, list_memories
        >>> configure({"api_key": "mv2_your_api_key"})
        >>> memories = list_memories()
        >>> for mem in memories:
        ...     print(f"{mem['display_name']} ({mem['id']})")
    """
    import urllib.request
    import urllib.error
    import urllib.parse
    import json

    # Resolve API key
    effective_api_key = (
        api_key
        or _global_config.get("api_key")
        or os.environ.get("MEMVID_API_KEY")
    )
    if not effective_api_key:
        raise ApiKeyRequiredError(
            "API key required for list_memories(). "
            "Set via configure({'api_key': 'mv2_...'}), MEMVID_API_KEY env var, or api_key parameter. "
            "Get your API key at https://memvid.com/dashboard/api-keys"
        )

    # Resolve dashboard URL
    effective_dashboard_url = (
        dashboard_url
        or _global_config.get("dashboard_url")
        or os.environ.get("MEMVID_DASHBOARD_URL")
        or "https://memvid.com"
    ).rstrip("/")

    url = f"{effective_dashboard_url}/api/memories"
    if project_id:
        url += f"?projectId={urllib.parse.quote(project_id)}"

    req = urllib.request.Request(url, method="GET")
    req.add_header("x-api-key", effective_api_key)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            response_data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8") if e.fp else ""
        raise MemvidError(f"Failed to list memories: {e.code} {e.reason}. {body_text}")
    except Exception as e:
        raise MemvidError(f"Failed to list memories: {e}")

    result_data = response_data.get("data", response_data)
    memories = result_data.get("memories", [])

    return [
        MemoryInfo(
            id=m.get("id"),
            organisation_id=m.get("organisation_id"),
            project_id=m.get("project_id"),
            slug=m.get("slug"),
            display_name=m.get("display_name"),
            description=m.get("description"),
            capacity_bytes=m.get("capacity_bytes"),
            created_at=m.get("created_at"),
            updated_at=m.get("updated_at"),
        )
        for m in memories
    ]


# ===========================================================================
# Project Management (Cloud)
# ===========================================================================


def create_project(
    name: str,
    *,
    description: Optional[str] = None,
    api_key: Optional[str] = None,
    dashboard_url: Optional[str] = None,
) -> CreateProjectResult:
    """Create a new project on the Memvid cloud.

    Projects are containers that organize multiple memories together.
    Use projects to group related memories (e.g., by environment, team, or use case).

    Args:
        name: Name for the project.
        description: Optional description.
        api_key: API key (uses global config or env var if not specified).
        dashboard_url: Dashboard URL (uses global config or env var if not specified).

    Returns:
        The created project info including the ID.

    Example:
        >>> from memvid_sdk import configure, create_project, create_memory
        >>> configure({
        ...     "api_key": "mv2_your_api_key",
        ...     "dashboard_url": "http://localhost:3001",
        ... })
        >>> project = create_project("Production Environment", description="All production memories")
        >>> print(f"Created project: {project['id']}")
        >>> # Create a memory in this project
        >>> memory = create_memory("Customer Support KB", project_id=project["id"])
    """
    import urllib.request
    import urllib.error
    import json

    # Resolve API key
    effective_api_key = (
        api_key
        or _global_config.get("api_key")
        or os.environ.get("MEMVID_API_KEY")
    )
    if not effective_api_key:
        raise MemvidError(
            "API key required for create_project(). "
            "Set via configure({'api_key': 'mv2_...'}), MEMVID_API_KEY env var, or api_key parameter. "
            "Get your API key at https://memvid.com/dashboard/api-keys"
        )

    # Resolve dashboard URL
    effective_dashboard_url = (
        dashboard_url
        or _global_config.get("dashboard_url")
        or os.environ.get("MEMVID_DASHBOARD_URL")
        or "https://memvid.com"
    ).rstrip("/")

    url = f"{effective_dashboard_url}/api/projects"

    body: Dict[str, Any] = {"name": name}
    if description:
        body["description"] = description

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("x-api-key", effective_api_key)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            response_data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8") if e.fp else ""
        raise MemvidError(f"Failed to create project: {e.code} {e.reason}. {body_text}")
    except Exception as e:
        raise MemvidError(f"Failed to create project: {e}")

    result_data = response_data.get("data", response_data)

    return CreateProjectResult(
        id=result_data.get("id"),
        organisation_id=result_data.get("organisation_id"),
        slug=result_data.get("slug"),
        name=result_data.get("name"),
        description=result_data.get("description"),
        created_at=result_data.get("created_at"),
        updated_at=result_data.get("updated_at"),
    )


def list_projects(
    *,
    api_key: Optional[str] = None,
    dashboard_url: Optional[str] = None,
) -> List[ProjectInfo]:
    """List all projects in your organisation.

    Args:
        api_key: API key (uses global config or env var if not specified).
        dashboard_url: Dashboard URL (uses global config or env var if not specified).

    Returns:
        List of project info dictionaries.

    Example:
        >>> from memvid_sdk import configure, list_projects
        >>> configure({"api_key": "mv2_your_api_key"})
        >>> projects = list_projects()
        >>> for proj in projects:
        ...     print(f"{proj['name']} ({proj['id']})")
    """
    import urllib.request
    import urllib.error
    import json

    # Resolve API key
    effective_api_key = (
        api_key
        or _global_config.get("api_key")
        or os.environ.get("MEMVID_API_KEY")
    )
    if not effective_api_key:
        raise MemvidError(
            "API key required for list_projects(). "
            "Set via configure({'api_key': 'mv2_...'}), MEMVID_API_KEY env var, or api_key parameter. "
            "Get your API key at https://memvid.com/dashboard/api-keys"
        )

    # Resolve dashboard URL
    effective_dashboard_url = (
        dashboard_url
        or _global_config.get("dashboard_url")
        or os.environ.get("MEMVID_DASHBOARD_URL")
        or "https://memvid.com"
    ).rstrip("/")

    url = f"{effective_dashboard_url}/api/projects"

    req = urllib.request.Request(url, method="GET")
    req.add_header("x-api-key", effective_api_key)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            response_data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8") if e.fp else ""
        raise MemvidError(f"Failed to list projects: {e.code} {e.reason}. {body_text}")
    except Exception as e:
        raise MemvidError(f"Failed to list projects: {e}")

    result_data = response_data.get("data", response_data)
    projects = result_data.get("projects", [])

    return [
        ProjectInfo(
            id=p.get("id"),
            organisation_id=p.get("organisation_id"),
            slug=p.get("slug"),
            name=p.get("name"),
            description=p.get("description"),
            created_at=p.get("created_at"),
            updated_at=p.get("updated_at"),
        )
        for p in projects
    ]


# ===========================================================================


class Memvid:
    """High-level facade over the compiled memvid-core handle.

    Supports context manager protocol for automatic resource cleanup:

    Example:
        >>> with memvid_sdk.use("basic", "data.mv2") as mem:
        ...     mem.put("Title", "label", {}, text="content")
        ...     results = mem.find("query")
        ... # File handle automatically closed

    Or for more control:
        >>> mem = memvid_sdk.use("basic", "data.mv2")
        >>> try:
        ...     mem.put("Title", "label", {}, text="content")
        ... finally:
        ...     mem.close()
    """

    def __init__(
        self,
        *,
        kind: str,
        core: _MemvidCore,
        attachments: Mapping[str, Any],
        memvid_api_key: Optional[str] = None,
    ):
        self._kind = kind
        self._core = core
        self._closed = False
        self._memvid_api_key = memvid_api_key
        self.tools = attachments.get(
            "tools", NoOp(f"kind '{kind}' did not register tools", f"memvid.{kind}.tools")
        )
        self.functions = attachments.get("functions", [])
        self.nodes = attachments.get(
            "nodes", NoOp(f"kind '{kind}' did not register nodes", f"memvid.{kind}.nodes")
        )
        self.as_query_engine = attachments.get("as_query_engine")

    def __enter__(self) -> "Memvid":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, closing the handle."""
        self.close()
        return None  # Don't suppress exceptions

    @property
    def path(self) -> str:
        return self._core.path()

    def put(
        self,
        title: Optional[str] = None,
        label: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        *,
        text: Optional[str] = None,
        file: Optional[str] = None,
        uri: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        labels: Optional[Sequence[str]] = None,
        kind: Optional[str] = None,
        track: Optional[str] = None,
        search_text: Optional[str] = None,
        timestamp: Optional[Union[int, str]] = None,
        enable_embedding: Optional[bool] = None,
        embedding_model: Optional[str] = None,
        auto_tag: bool = True,
        extract_dates: bool = True,
        vector_compression: bool = False,
    ) -> str:
        # Set vector compression mode if requested
        if vector_compression:
            self._core.set_vector_compression(True)

        # Default title from filename if file provided
        if title is None:
            if file:
                title = Path(file).stem
            else:
                title = "Untitled"

        # Default label from file extension or "text"
        if label is None:
            if file:
                label = Path(file).suffix.lstrip(".") or "file"
            else:
                label = "text"

        payload: MutableMapping[str, Any] = {
            "title": title,
            "label": label,
            "metadata": dict(metadata) if metadata else {},
            "auto_tag": auto_tag,
            "extract_dates": extract_dates,
        }
        # Only include enable_embedding if explicitly set (None = auto-detect via OPENAI_API_KEY)
        if enable_embedding is not None:
            payload["enable_embedding"] = enable_embedding
        if text is not None:
            payload["text"] = text
        if file is not None:
            payload["file"] = file
        if uri is not None:
            payload["uri"] = uri
        if tags is not None:
            payload["tags"] = list(tags)
        merged_labels = list(labels or [])
        if label not in merged_labels:
            merged_labels.insert(0, label)
        payload["labels"] = merged_labels
        if kind is not None:
            payload["kind"] = kind
        if track is not None:
            payload["track"] = track
        if search_text is not None:
            payload["search_text"] = search_text
        if timestamp is not None:
            payload["timestamp"] = timestamp
        if embedding_model is not None:
            payload["embedding_model"] = embedding_model
        frame_id = self._core.put(payload)
        track_command(self.path, "put", True)
        return str(frame_id)

    def put_many(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        embeddings: Optional[Sequence[Sequence[float]]] = None,
        embedder: Optional["embeddings.EmbeddingProvider"] = None,
        embedding_identity: Optional[Mapping[str, Any]] = None,
        opts: Optional[Mapping[str, Any]] = None,
    ) -> List[str]:
        """
        Batch ingestion of multiple documents in a single operation.

        Eliminates Python FFI overhead by processing all documents in Rust,
        providing 100x+ speedup compared to individual put() calls.

        Args:
            requests: List of document dictionaries, each containing:
                - title (str, required): Document title
                - label (str, required): Primary label/category
                - text (str, required): Document content
                - uri (str, optional): Document URI
                - metadata (dict, optional): Key-value metadata
                - tags (list[str], optional): List of tags
                - labels (list[str], optional): Additional labels

            opts: Optional dict with batch operation settings:
                - compression_level (int): 0=none, 1=fast, 3=default, 11=max (default: 3)
                - disable_auto_checkpoint (bool): Skip auto-checkpoint during batch (default: True)
                - skip_sync (bool): Skip fsync for maximum speed (default: False)
                - enable_embedding (bool): Generate embeddings (default: False)
                - auto_tag (bool): Auto-extract tags (default: False)
                - extract_dates (bool): Extract dates from content (default: False)

        Returns:
            List of frame IDs as strings

        Example:
            >>> docs = [
            ...     {"title": "Doc 1", "label": "news", "text": "First document..."},
            ...     {"title": "Doc 2", "label": "news", "text": "Second document..."},
            ... ]
            >>> frame_ids = mem.put_many(docs)
            >>> print(f"Ingested {len(frame_ids)} documents")
        """
        if not requests:
            return []

        if embedder is not None and embeddings is not None:
            raise ValueError("Pass either embeddings=... or embedder=..., not both")

        if embedder is not None and embeddings is None:
            texts = [str(req.get("text", "")) for req in requests]
            embeddings = embedder.embed_documents(texts)
            if embedding_identity is None:
                embedding_identity = _embedding_identity_from_embedder(embedder)

        # Validate each request has required fields
        for i, req in enumerate(requests):
            if not isinstance(req, Mapping):
                raise ValueError(f"Request {i} must be a dict, got {type(req)}")
            if "title" not in req:
                raise ValueError(f"Request {i} missing required field 'title'")
            if "label" not in req:
                raise ValueError(f"Request {i} missing required field 'label'")
            if "text" not in req:
                raise ValueError(f"Request {i} missing required field 'text'")

        # Convert requests to list of dicts (ensure mutable for FFI)
        validated_requests = []
        for req in requests:
            doc = {
                "title": req["title"],
                "label": req["label"],
                "text": req["text"],
            }
            if "uri" in req and req["uri"] is not None:
                doc["uri"] = req["uri"]
            if "metadata" in req:
                doc["metadata"] = dict(req["metadata"])
            else:
                doc["metadata"] = {}
            if "tags" in req:
                doc["tags"] = list(req["tags"])
            else:
                doc["tags"] = []
            if "labels" in req:
                doc["labels"] = list(req["labels"])
            else:
                doc["labels"] = []
            validated_requests.append(doc)

        embedding_dimension: Optional[int] = None
        if embeddings is not None:
            if len(embeddings) != len(validated_requests):
                raise ValueError(
                    f"Embeddings length ({len(embeddings)}) must match requests length ({len(validated_requests)})"
                )
            if embeddings:
                embedding_dimension = len(embeddings[0])
                for i, vec in enumerate(embeddings):
                    if len(vec) != embedding_dimension:
                        raise ValueError(
                            f"Embeddings must have consistent dimension (expected {embedding_dimension}, got {len(vec)} at index {i})"
                        )

        if embedding_identity is not None and embeddings is not None:
            for doc in validated_requests:
                meta = doc.get("metadata")
                if not isinstance(meta, dict):
                    meta = {}
                    doc["metadata"] = meta
                _apply_embedding_identity_metadata(meta, embedding_identity, embedding_dimension)

        # Prepare options dict for FFI
        opts_dict = {}
        if opts:
            if "compression_level" in opts:
                opts_dict["compression_level"] = opts["compression_level"]
            if "disable_auto_checkpoint" in opts:
                opts_dict["disable_auto_checkpoint"] = opts["disable_auto_checkpoint"]
            if "skip_sync" in opts:
                opts_dict["skip_sync"] = opts["skip_sync"]
            if "enable_embedding" in opts:
                opts_dict["enable_embedding"] = opts["enable_embedding"]
            if "embedding_model" in opts:
                opts_dict["embedding_model"] = opts["embedding_model"]
            if "auto_tag" in opts:
                opts_dict["auto_tag"] = opts["auto_tag"]
            if "extract_dates" in opts:
                opts_dict["extract_dates"] = opts["extract_dates"]

        # Call FFI binding (GIL will be released in Rust)
        frame_ids = self._core.put_many(validated_requests, embeddings, opts_dict if opts_dict else None)

        # Convert to strings
        return [str(fid) for fid in frame_ids]

    def remove(self, frame_id: Union[int, str]) -> int:
        """
        Remove a frame by its ID.

        This performs a soft delete - the frame is marked as deleted and removed
        from search indexes, but remains in the file for audit purposes.

        Args:
            frame_id: The frame ID to remove (returned by put())

        Returns:
            The sequence number of the deletion operation

        Example:
            >>> frame_id = mem.put(title="Doc", text="content")
            >>> mem.remove(frame_id)  # Remove the frame
        """
        # Convert string frame_id to int if needed
        fid = int(frame_id) if isinstance(frame_id, str) else frame_id
        seq = self._core.remove(fid)
        track_command(self.path, "remove", True)
        return seq

    def correct(
        self,
        statement: str,
        *,
        topics: Optional[List[str]] = None,
        source: Optional[str] = None,
        boost: float = 2.0,
    ) -> str:
        """
        Store a correction with retrieval priority boost.

        Corrections are stored as frames with special metadata that gives them
        priority during retrieval. This ensures corrected information surfaces
        first when relevant queries are made.

        Args:
            statement: The correction statement (e.g., "Ben reported to Chloe before 2025")
            topics: Optional list of topics for better retrieval matching
            source: Optional source attribution for the correction
            boost: Retrieval boost factor (default: 2.0)

        Returns:
            Frame ID of the stored correction

        Example:
            >>> mv.correct("Ben Koenig reported to Chloe Nguyen before 2025")

            >>> mv.correct(
            ...     "Ben Koenig reported to Chloe Nguyen before 2025",
            ...     topics=["Ben Koenig", "manager", "reporting"],
            ...     source="HR clarification"
            ... )
        """
        import uuid

        # Build metadata for correction
        metadata = {
            "memvid.correction": "true",
            "memvid.correction.boost": str(boost),
        }
        if source:
            metadata["memvid.correction.source"] = source

        # Truncate title if needed
        max_len = 50
        title_text = statement if len(statement) <= max_len else statement[:max_len - 3] + "..."

        payload: MutableMapping[str, Any] = {
            "title": f"Correction: {title_text}",
            "label": "correction",
            "text": statement,
            "uri": f"mv2://correction/{uuid.uuid4()}",
            "labels": ["correction"],
            "tags": topics or [],
            "metadata": metadata,
            "enable_embedding": False,
            "auto_tag": False,
            "extract_dates": False,
        }

        frame_id = self._core.put(payload)
        track_command(self.path, "correct", True)
        return str(frame_id)

    def correct_many(
        self,
        corrections: Sequence[Mapping[str, Any]],
    ) -> List[str]:
        """
        Store multiple corrections with retrieval priority boost.

        Args:
            corrections: List of correction dictionaries, each containing:
                - statement (str, required): The correction statement
                - topics (list[str], optional): Topics for better retrieval matching
                - source (str, optional): Source attribution
                - boost (float, optional): Retrieval boost factor (default: 2.0)

        Returns:
            List of frame IDs for the stored corrections

        Example:
            >>> mv.correct_many([
            ...     {"statement": "Ben reported to Chloe before 2025"},
            ...     {"statement": "Ava was promoted in January 2025", "topics": ["Ava", "promotion"]}
            ... ])
        """
        frame_ids = []
        for corr in corrections:
            if not isinstance(corr, Mapping):
                raise ValueError(f"Each correction must be a dict, got {type(corr)}")
            if "statement" not in corr:
                raise ValueError("Each correction must have a 'statement' field")

            frame_id = self.correct(
                corr["statement"],
                topics=corr.get("topics"),
                source=corr.get("source"),
                boost=corr.get("boost", 2.0),
            )
            frame_ids.append(frame_id)

        track_command(self.path, "correct_many", True)
        return frame_ids

    def find(
        self,
        query: str,
        *,
        k: int = 10,
        snippet_chars: int = 480,
        scope: Optional[str] = None,
        cursor: Optional[str] = None,
        mode: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        query_embedding_model: Optional[str] = None,
        adaptive: Optional[bool] = None,
        min_relevancy: Optional[float] = None,
        max_k: Optional[int] = None,
        adaptive_strategy: Optional[str] = None,
        embedder: Optional["embeddings.EmbeddingProvider"] = None,
        as_of_frame: Optional[int] = None,
        as_of_ts: Optional[int] = None,
        acl_context: Optional[Mapping[str, Any]] = None,
        acl_enforcement_mode: str = "audit",
    ) -> FindResult:
        # Track query usage if API key is configured
        api_key = self._memvid_api_key or _global_config.get("api_key") or os.environ.get("MEMVID_API_KEY")
        if api_key:
            _track_query_usage(api_key, 1)

        if embedder is not None and query_embedding is not None:
            raise ValueError("Pass either query_embedding=... or embedder=..., not both")
        if embedder is not None and query_embedding is None and (mode or "auto") != "lex":
            query_embedding = embedder.embed_query(query)

        # Auto-detect mode: use semantic search if vectors exist and OPENAI_API_KEY is available
        effective_mode = mode
        if effective_mode is None or effective_mode == "auto":
            stats = self._core.stats()
            has_vectors = stats.get("vec_index_bytes", 0) > 100
            has_openai_key = bool(os.environ.get("OPENAI_API_KEY", "").strip())
            # Use semantic if we have vectors and can embed the query
            if has_vectors and (has_openai_key or query_embedding is not None or embedder is not None):
                effective_mode = "sem"
                # Auto-set query_embedding_model for OpenAI if not provided
                if query_embedding is None and embedder is None and query_embedding_model is None:
                    query_embedding_model = "openai-small"
            else:
                effective_mode = "lex"

        result = self._core.find(
            query,
            k=k,
            snippet_chars=snippet_chars,
            scope=scope,
            cursor=cursor,
            mode=effective_mode,
            query_embedding=query_embedding,
            query_embedding_model=query_embedding_model,
            adaptive=adaptive,
            min_relevancy=min_relevancy,
            max_k=max_k,
            adaptive_strategy=adaptive_strategy,
            as_of_frame=as_of_frame,
            as_of_ts=as_of_ts,
            acl_context=acl_context,
            acl_enforcement_mode=acl_enforcement_mode,
        )
        track_command(self.path, "find", True)
        return result

    def ask(
        self,
        question: str,
        *,
        k: int = 6,
        mode: str = "auto",
        snippet_chars: int = 320,
        scope: Optional[str] = None,
        since: Optional[int] = None,
        until: Optional[int] = None,
        context_only: bool = False,
        query_embedding: Optional[List[float]] = None,
        query_embedding_model: Optional[str] = None,
        adaptive: Optional[bool] = None,
        min_relevancy: Optional[float] = None,
        max_k: Optional[int] = None,
        adaptive_strategy: Optional[str] = None,
        embedder: Optional["embeddings.EmbeddingProvider"] = None,
        model: Optional[str] = None,
        llm_context_chars: Optional[int] = None,
        api_key: Optional[str] = None,
        mask_pii: bool = False,
        return_sources: bool = False,
        show_chunks: bool = False,
        acl_context: Optional[Mapping[str, Any]] = None,
        acl_enforcement_mode: str = "audit",
    ) -> AskResult:
        # Track query usage if API key is configured
        memvid_key = self._memvid_api_key or _global_config.get("api_key") or os.environ.get("MEMVID_API_KEY")
        if memvid_key:
            _track_query_usage(memvid_key, 1)

        if embedder is not None and query_embedding is not None:
            raise ValueError("Pass either query_embedding=... or embedder=..., not both")
        if embedder is not None and query_embedding is None and mode != "lex":
            query_embedding = embedder.embed_query(question)

        # Auto-detect mode: use semantic search if vectors exist and OPENAI_API_KEY is available
        effective_mode = mode
        if effective_mode is None or effective_mode == "auto":
            stats = self._core.stats()
            has_vectors = stats.get("vec_index_bytes", 0) > 100
            has_openai_key = bool(os.environ.get("OPENAI_API_KEY", "").strip())
            # Use semantic if we have vectors and can embed the query
            if has_vectors and (has_openai_key or query_embedding is not None or embedder is not None):
                effective_mode = "sem"
                # Auto-set query_embedding_model for OpenAI if not provided
                if query_embedding is None and embedder is None and query_embedding_model is None:
                    query_embedding_model = "openai-small"
            else:
                effective_mode = "lex"

        # Auto-detect LLM model from available API keys if not specified and not context_only
        effective_model = model
        if effective_model is None and not context_only:
            # Check for available LLM API keys in order of preference
            if os.environ.get("GEMINI_API_KEY", "").strip():
                effective_model = "gemini"
            elif os.environ.get("OPENAI_API_KEY", "").strip():
                effective_model = "openai"
            elif os.environ.get("ANTHROPIC_API_KEY", "").strip() or os.environ.get("CLAUDE_API_KEY", "").strip():
                effective_model = "claude"
            elif os.environ.get("GROQ_API_KEY", "").strip():
                effective_model = "groq"
            elif os.environ.get("XAI_API_KEY", "").strip():
                effective_model = "xai"
            elif os.environ.get("MISTRAL_API_KEY", "").strip():
                effective_model = "mistral"
            # If no LLM key found, effective_model stays None and ask() returns context only

        response = self._core.ask(
            question,
            k=k,
            mode=effective_mode,
            snippet_chars=snippet_chars,
            scope=scope,
            since=since,
            until=until,
            context_only=context_only,
            query_embedding=query_embedding,
            query_embedding_model=query_embedding_model,
            adaptive=adaptive,
            min_relevancy=min_relevancy,
            max_k=max_k,
            adaptive_strategy=adaptive_strategy,
            model=effective_model,
            llm_context_chars=llm_context_chars,
            api_key=api_key,
            return_sources=return_sources,
            show_chunks=show_chunks,
            acl_context=acl_context,
            acl_enforcement_mode=acl_enforcement_mode,
        )

        # Apply PII masking if requested
        if mask_pii:
            from ._lib import mask_pii as _mask_pii_fn

            # Mask the aggregated context
            if "context" in response:
                response["context"] = _mask_pii_fn(response["context"])

            # Mask the answer
            if "answer" in response and response["answer"]:
                response["answer"] = _mask_pii_fn(response["answer"])

            # Mask answer_lines
            if "answer_lines" in response and response["answer_lines"]:
                response["answer_lines"] = [_mask_pii_fn(line) for line in response["answer_lines"]]

            # Mask text in each hit
            if "hits" in response:
                for hit in response["hits"]:
                    if "text" in hit:
                        hit["text"] = _mask_pii_fn(hit["text"])
                    if "chunk_text" in hit and hit["chunk_text"]:
                        hit["chunk_text"] = _mask_pii_fn(hit["chunk_text"])
                    if "snippet" in hit and hit["snippet"]:
                        hit["snippet"] = _mask_pii_fn(hit["snippet"])
                    if "tags" in hit and hit["tags"]:
                        hit["tags"] = [_mask_pii_fn(tag) for tag in hit["tags"]]
                    if "labels" in hit and hit["labels"]:
                        hit["labels"] = [_mask_pii_fn(label) for label in hit["labels"]]

            # Mask context_fragments
            if "context_fragments" in response:
                for fragment in response["context_fragments"]:
                    if "text" in fragment and fragment["text"]:
                        fragment["text"] = _mask_pii_fn(fragment["text"])

            # Mask sources
            if "sources" in response:
                for source in response["sources"]:
                    if "snippet" in source and source["snippet"]:
                        source["snippet"] = _mask_pii_fn(source["snippet"])
                    if "tags" in source and source["tags"]:
                        source["tags"] = [_mask_pii_fn(tag) for tag in source["tags"]]
                    if "labels" in source and source["labels"]:
                        source["labels"] = [_mask_pii_fn(label) for label in source["labels"]]

            # Mask primary_source
            if "primary_source" in response and response["primary_source"]:
                ps = response["primary_source"]
                if "snippet" in ps and ps["snippet"]:
                    ps["snippet"] = _mask_pii_fn(ps["snippet"])
                if "tags" in ps and ps["tags"]:
                    ps["tags"] = [_mask_pii_fn(tag) for tag in ps["tags"]]
                if "labels" in ps and ps["labels"]:
                    ps["labels"] = [_mask_pii_fn(label) for label in ps["labels"]]

        # Calculate grounding score
        response = self._verify_grounding(response)

        # Build follow-up suggestions if confidence is low
        response = self._build_follow_up(response)

        return response

    def _verify_grounding(self, response: AskResult) -> AskResult:
        """Verify how well the answer is grounded in the provided context."""
        import re

        answer = response.get("answer", "") or ""
        context = response.get("context", "") or ""

        # Empty answer = no hallucination
        if not answer:
            response["grounding"] = {
                "score": 1.0,
                "label": "HIGH",
                "sentence_count": 0,
                "grounded_sentences": 0,
                "has_warning": False,
                "warning_reason": None,
            }
            return response

        # No context = potential hallucination
        if not context:
            response["grounding"] = {
                "score": 0.0,
                "label": "LOW",
                "sentence_count": 1,
                "grounded_sentences": 0,
                "has_warning": True,
                "warning_reason": "No context provided - answer may be hallucinated",
            }
            return response

        # Normalize context for comparison
        context_lower = context.lower()
        context_words = set(w for w in re.split(r'[^a-zA-Z0-9]+', context_lower) if len(w) > 2)

        # Split answer into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]', answer) if s.strip() and len(s.strip()) > 10]

        if not sentences:
            response["grounding"] = {
                "score": 0.5,
                "label": "MEDIUM",
                "sentence_count": 0,
                "grounded_sentences": 0,
                "has_warning": False,
                "warning_reason": None,
            }
            return response

        sentence_scores = []
        grounded_count = 0

        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence_words = set(w for w in re.split(r'[^a-zA-Z0-9]+', sentence_lower) if len(w) > 2)

            if not sentence_words:
                sentence_scores.append(0.5)
                continue

            # Calculate word overlap
            overlap = len(sentence_words & context_words)
            score = overlap / max(len(sentence_words), 1)

            # Add phrase bonus for exact matches
            phrase_bonus = 0.0
            if sentence_lower in context_lower:
                phrase_bonus = 0.3
            else:
                # Check for significant substring matches
                words = sentence_lower.split()
                if len(words) >= 3:
                    phrase = " ".join(words[:3])
                    if phrase in context_lower:
                        phrase_bonus = 0.15

            final_score = min(score + phrase_bonus, 1.0)
            sentence_scores.append(final_score)

            if final_score >= 0.3:
                grounded_count += 1

        overall_score = sum(sentence_scores) / len(sentence_scores) if sentence_scores else 0.5

        # Determine warning
        has_warning = False
        warning_reason = None
        if overall_score < 0.2:
            has_warning = True
            warning_reason = "Answer appears to be poorly grounded in context"
        elif overall_score < 0.4 and grounded_count < len(sentences) // 2:
            has_warning = True
            warning_reason = "Some statements may not be supported by context"

        # Determine label
        if overall_score >= 0.7:
            label = "HIGH"
        elif overall_score >= 0.4:
            label = "MEDIUM"
        else:
            label = "LOW"

        response["grounding"] = {
            "score": overall_score,
            "label": label,
            "sentence_count": len(sentences),
            "grounded_sentences": grounded_count,
            "has_warning": has_warning,
            "warning_reason": warning_reason,
        }

        return response

    def _build_follow_up(self, response: AskResult) -> AskResult:
        """Add follow-up suggestions if answer confidence is low."""
        hits = response.get("hits", [])

        # Check if retrieval has no results
        no_hits = len(hits) == 0

        # Check if retrieval scores are very low (CLI-style log scores only)
        very_low_retrieval = False
        if hits:
            first_score = hits[0].get("score")
            # Only trigger for clearly negative log scores (CLI format)
            if first_score is not None and first_score < -2.0:
                very_low_retrieval = True

        # Check grounding if available (from model inference)
        grounding = response.get("grounding", {})
        has_grounding_warning = grounding.get("has_warning", False)
        low_grounding = grounding.get("score", 1.0) < 0.3 if grounding else False

        # Detect "Not enough information" type answers (primary trigger)
        answer = response.get("answer", "") or ""
        no_info_phrases = [
            "not enough information",
            "no relevant information",
            "cannot find",
            "no direct synthesis",
            "i don't have",
            "i cannot provide",
            "unable to find",
            "no information available",
        ]
        no_info_answer = any(phrase in answer.lower() for phrase in no_info_phrases)

        # Trigger follow-up if:
        # 1. Answer indicates "not enough info" (primary signal)
        # 2. Grounding score is low or has warning
        # 3. No retrieval hits at all
        # 4. Very low retrieval scores (CLI format)
        needs_followup = no_info_answer or has_grounding_warning or low_grounding or no_hits or very_low_retrieval

        if not needs_followup:
            return response

        # Get available topics from timeline
        try:
            timeline_entries = self._core.timeline(limit=20, reverse=False)
            available_topics: List[str] = []
            seen: set = set()
            for entry in timeline_entries:
                preview = entry.get("preview", "").strip()
                if not preview or len(preview) < 5:
                    continue
                # Get first line and truncate
                first_line = preview.split("\n")[0]
                if len(first_line) > 60:
                    first_line = first_line[:57] + "..."
                if first_line not in seen:
                    seen.add(first_line)
                    available_topics.append(first_line)
                if len(available_topics) >= 5:
                    break
        except Exception:
            available_topics = []

        # Determine reason based on what triggered follow-up
        if no_hits or very_low_retrieval:
            reason = "No relevant information found in memory"
        elif has_grounding_warning or low_grounding:
            reason = "Answer may not be well-supported by the available context"
        elif no_info_answer:
            reason = "The model indicated it could not find relevant information"
        else:
            reason = "Low confidence in the answer"

        # Generate suggestions
        if available_topics:
            suggestions = [f"Tell me about {topic}" for topic in available_topics[:3]]
            suggestions.append("What topics are in this memory?")
            hint = "This memory contains information about different topics. Try asking about those instead."
        else:
            suggestions = [
                "What information is stored in this memory?",
                "Can you list the main topics covered?",
            ]
            hint = "This memory may not contain information about your query."

        response["follow_up"] = {
            "needed": True,
            "reason": reason,
            "hint": hint,
            "available_topics": available_topics,
            "suggestions": suggestions,
        }

        return response

    def timeline(
        self,
        *,
        limit: int = 100,
        since: Optional[int] = None,
        until: Optional[int] = None,
        reverse: bool = False,
        as_of_frame: Optional[int] = None,
        as_of_ts: Optional[int] = None,
    ) -> List[TimelineEntry]:
        """Query the timeline with optional Replay filters.

        Args:
            limit: Maximum number of entries to return
            since: Filter entries with timestamp >= since (Unix timestamp)
            until: Filter entries with timestamp <= until (Unix timestamp)
            reverse: Return entries in reverse chronological order
            as_of_frame: Replay filter - only show frames with ID <= as_of_frame
            as_of_ts: Replay filter - only show frames with timestamp <= as_of_ts

        Returns:
            List of timeline entries
        """
        return self._core.timeline(
            limit=limit,
            since=since,
            until=until,
            reverse=reverse,
            as_of_frame=as_of_frame,
            as_of_ts=as_of_ts,
        )

    def stats(self) -> StatsResult:
        """Get memory statistics including capacity, frame count, and index status."""
        result = self._core.stats()
        track_command(self.path, "stats", True)
        return result

    def seal(self) -> None:
        self._core.seal()

    def rebuild_time_index(self) -> None:
        """
        Rebuild the time index. Call this after using put_many() if you need
        time-based queries (like ask() with temporal context).
        """
        self.doctor(rebuild_time_index=True, quiet=True)

    def put_file(
        self,
        file_path: str,
        *,
        label: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        embedder: Optional["embeddings.EmbeddingProvider"] = None,
        enable_embedding: Optional[bool] = None,
        embedding_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a document file (PDF, XLSX, PPTX, DOCX) with automatic parsing.

        Uses fast batch ingestion internally. Call rebuild_time_index() after
        seal() if you need time-based queries.

        Args:
            file_path: Path to the document file
            label: Label for all frames (default: "document")
            metadata: Additional metadata for all frames
            embedder: External embedding provider (OpenAI, Cohere, etc.)
            enable_embedding: Enable internal embeddings (None = auto-detect via OPENAI_API_KEY)
            embedding_model: Internal embedding model name

        Returns:
            Dict with frames_added, type, and filename
        """
        from .documents import parse, get_document_type

        filename = os.path.basename(file_path)
        doc_type = get_document_type(file_path)
        label = label or "document"
        base_metadata = dict(metadata) if metadata else {}

        # Try parsing
        result = parse(file_path)

        # PDF fallback: if parse returns None, use Rust core's built-in pdf_extract
        if result is None and doc_type == "pdf":
            print(f"[memvid] Using Rust pdf_extract for {filename}")
            self.put(
                file=file_path,
                label=label,
                metadata={
                    **base_metadata,
                    "doc_name": filename,
                    "doc_type": "pdf",
                    "fallback": "rust_pdf_extract",
                },
                enable_embedding=enable_embedding,
                embedding_model=embedding_model,
            )
            return {"frames_added": 1, "type": "pdf", "filename": filename}

        if result is None:
            raise RuntimeError(f"Failed to parse document: {filename}")

        # Build items for batch processing
        items: List[Dict[str, Any]] = []

        for item in result["items"]:
            if result["type"] == "pdf":
                title = f"{result['filename']} [Page {item['number']}]"
                item_metadata = {
                    **base_metadata,
                    "doc_name": result["filename"],
                    "doc_type": result["type"],
                    "page_number": item["number"],
                    "total_pages": result["total_items"],
                }
            elif result["type"] == "xlsx":
                title = f"{result['filename']} [Sheet: {item.get('name', item['number'])}]"
                item_metadata = {
                    **base_metadata,
                    "doc_name": result["filename"],
                    "doc_type": result["type"],
                    "sheet_name": item.get("name"),
                    "sheet_index": item["number"],
                    "total_sheets": result["total_items"],
                }
            elif result["type"] == "pptx":
                title = f"{result['filename']} [Slide {item['number']}]"
                item_metadata = {
                    **base_metadata,
                    "doc_name": result["filename"],
                    "doc_type": result["type"],
                    "slide_number": item["number"],
                    "slide_title": item.get("title"),
                    "total_slides": result["total_items"],
                }
            else:  # docx
                title = result["filename"]
                item_metadata = {
                    **base_metadata,
                    "doc_name": result["filename"],
                    "doc_type": result["type"],
                }

            items.append({
                "title": title,
                "label": label,
                "text": item["text"],
                "metadata": item_metadata,
            })

        # Use put_many for fast batch ingestion
        # Pass enable_embedding only if explicitly set (None = auto-detect in put_many)
        opts = {}
        if enable_embedding is not None and not embedder:
            opts["enable_embedding"] = enable_embedding
        if embedding_model and not embedder:
            opts["embedding_model"] = embedding_model

        self.put_many(items, embedder=embedder, opts=opts if opts else None)

        return {
            "frames_added": len(items),
            "type": result["type"],
            "filename": result["filename"],
        }

    def put_files(
        self,
        dir_path: str,
        *,
        label: Optional[str] = None,
        extensions: Optional[List[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        embedder: Optional["embeddings.EmbeddingProvider"] = None,
        enable_embedding: Optional[bool] = None,
        embedding_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest multiple document files from a directory.

        Args:
            dir_path: Path to directory containing documents
            label: Label for all frames (default: "document")
            extensions: File extensions to include (default: all supported)
            metadata: Additional metadata for all frames
            embedder: External embedding provider
            enable_embedding: Enable internal embeddings
            embedding_model: Internal embedding model name

        Returns:
            Dict with files_processed, frames_added, and files list
        """
        extensions = extensions or [".pdf", ".xlsx", ".xls", ".pptx", ".ppt", ".docx", ".doc"]

        files = [
            f for f in os.listdir(dir_path)
            if any(f.lower().endswith(ext) for ext in extensions)
        ]

        files_processed = 0
        frames_added = 0
        results: List[Dict[str, Any]] = []

        for file in files:
            file_path = os.path.join(dir_path, file)
            result = self.put_file(
                file_path,
                label=label,
                metadata=metadata,
                embedder=embedder,
                enable_embedding=enable_embedding,
                embedding_model=embedding_model,
            )
            files_processed += 1
            frames_added += result["frames_added"]
            results.append({
                "filename": result["filename"],
                "frames_added": result["frames_added"],
                "type": result["type"],
            })

        return {
            "files_processed": files_processed,
            "frames_added": frames_added,
            "files": results,
        }

    def commit(self) -> None:
        """Explicitly commit pending WAL/index changes."""
        self._core.commit()

    def commit_parallel(self, opts=None) -> None:
        """Commit using parallel build if available (no-op fallback if not compiled)."""
        from memvid_sdk._lib import BuildOpts
        commit_parallel = getattr(self._core, "commit_parallel", None)
        if commit_parallel is not None:
            if opts is None:
                opts = BuildOpts()
            commit_parallel(opts)
        else:
            self.commit()

    def frame(self, uri: str) -> Dict[str, Any]:
        return self._core.frame(uri)

    def blob(self, uri: str) -> bytes:
        return self._core.blob(uri)

    def close(self) -> None:
        """Close the file handle and release resources.

        Safe to call multiple times. After closing, most operations
        will raise RuntimeError.
        """
        if self._closed:
            return
        try:
            self._core.close()
        except RuntimeError:
            pass
        finally:
            self._closed = True

    @property
    def closed(self) -> bool:
        """True if the handle has been closed."""
        return self._closed

    def enable_lex(self) -> None:
        self._core.enable_lex()

    def enable_vec(self) -> None:
        self._core.enable_vec()

    def apply_ticket(self, ticket: str) -> None:
        self._core.apply_ticket(ticket)

    def get_memory_binding(self) -> Optional[Dict[str, Any]]:
        """Get the current memory binding, if any."""
        return self._core.get_memory_binding()

    def unbind_memory(self) -> None:
        """Unbind from the dashboard memory."""
        self._core.unbind_memory()

    def get_capacity(self) -> int:
        """Get the current capacity in bytes."""
        return self._core.get_capacity()

    def current_ticket(self) -> Dict[str, Any]:
        """Get the current ticket information."""
        return self._core.current_ticket()

    def sync_tickets(
        self,
        memory_id: str,
        api_key: str,
        api_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sync tickets from the API and apply to this file."""
        return self._core.sync_tickets(memory_id, api_key, api_url)

    def verify(self, path: Optional[str] = None, *, deep: bool = False) -> Dict[str, Any]:
        if _verify is None:
            raise RuntimeError("verify support not available in this build")
        if isinstance(self, Memvid):
            resolved = self.path if path is None else str(path)
        else:
            resolved = str(self)
        return _verify(resolved, deep=deep)

    def doctor(
        self,
        path: Optional[str] = None,
        *,
        rebuild_time_index: bool = False,
        rebuild_lex_index: bool = False,
        rebuild_vec_index: bool = False,
        vacuum: bool = False,
        dry_run: bool = False,
        quiet: bool = False,
    ) -> Dict[str, Any]:
        from ._lib import doctor as _doctor

        if isinstance(self, Memvid):
            resolved = self.path if path is None else str(path)
        else:
            resolved = str(self)
        return _doctor(
            resolved,
            rebuild_time_index=rebuild_time_index,
            rebuild_lex_index=rebuild_lex_index,
            rebuild_vec_index=rebuild_vec_index,
            vacuum=vacuum,
            dry_run=dry_run,
            quiet=quiet,
        )

    def put_pdf_tables(
        self,
        pdf_path: str,
        *,
        embed_rows: bool = True,
    ) -> Dict[str, Any]:
        """Extract tables from a PDF and store them in the memory.

        Args:
            pdf_path: Path to the PDF file
            embed_rows: If True, embed individual rows for semantic search

        Returns:
            Dict with extraction results including table_count and table_ids
        """
        return self._core.put_pdf_tables(pdf_path, embed_rows=embed_rows)

    def list_tables(self) -> List[Dict[str, Any]]:
        """List all tables stored in the memory.

        Returns:
            List of table metadata dicts with table_id, row_count, col_count, etc.
        """
        return self._core.list_tables()

    def get_table(
        self,
        table_id: str,
        *,
        format: str = "dict",
    ) -> Any:
        """Retrieve a table by ID.

        Args:
            table_id: The table identifier
            format: Output format - "dict" (default), "csv", or "json"

        Returns:
            Table data in the requested format
        """
        return self._core.get_table(table_id, format=format)

    # ─────────────────────────────────────────────────────────────────────────
    # Session Recording / Time-Travel Replay
    # ─────────────────────────────────────────────────────────────────────────

    def session_start(self, name: Optional[str] = None) -> str:
        """Start a new recording session.

        All subsequent operations (put, find, ask) will be recorded until
        session_end() is called. Sessions can be replayed later with different
        parameters for debugging or testing.

        Args:
            name: Optional descriptive name for the session

        Returns:
            Session ID (UUID string)

        Example:
            >>> session_id = mem.session_start("Debug Session")
            >>> mem.put("Title", "label", {}, text="content")
            >>> results = mem.find("query")
            >>> session = mem.session_end()
        """
        session_start_fn = getattr(self._core, "session_start", None)
        if session_start_fn is None:
            raise RuntimeError("session_start not available - replay feature not compiled")
        return session_start_fn(name)

    def session_end(self) -> Dict[str, Any]:
        """End the current recording session.

        Returns:
            Session summary dict with:
                - session_id: UUID string
                - name: Session name (if provided)
                - created_secs: Unix timestamp when session started
                - ended_secs: Unix timestamp when session ended
                - action_count: Number of recorded actions
                - checkpoint_count: Number of checkpoints
                - duration_secs: Total session duration
        """
        session_end_fn = getattr(self._core, "session_end", None)
        if session_end_fn is None:
            raise RuntimeError("session_end not available - replay feature not compiled")
        return session_end_fn()

    def session_list(self) -> List[Dict[str, Any]]:
        """List all recorded sessions.

        Returns:
            List of session summary dicts (same format as session_end)
        """
        session_list_fn = getattr(self._core, "session_list", None)
        if session_list_fn is None:
            raise RuntimeError("session_list not available - replay feature not compiled")
        return session_list_fn()

    def session_replay(
        self,
        session_id: str,
        *,
        top_k: Optional[int] = None,
        adaptive: Optional[bool] = None,
        audit: Optional[bool] = None,
        use_model: Optional[str] = None,
        diff: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Replay a recorded session with optional parameter overrides.

        This enables "time-travel debugging" and "audit mode" for compliance.

        Args:
            session_id: UUID of the session to replay
            top_k: Override top-k for search operations
            adaptive: Override adaptive retrieval setting
            audit: Enable audit mode - uses frozen retrieval for ASK actions.
                   When True, ASK actions use recorded frame IDs instead of
                   re-executing search, enabling deterministic replay.
            use_model: Override model for audit replay (format: "provider:model").
                       When set, re-executes LLM call with frozen context using
                       this model. Examples: "openai:gpt-4o", "groq:llama-3.3-70b-versatile",
                       "anthropic:claude-sonnet-4-20250514"
            diff: Generate diff report comparing original vs new answers.
                  Requires audit mode to be enabled.

        Returns:
            Replay result dict with:
                - total_actions: Total actions in session
                - matched_actions: Actions that matched original results
                - mismatched_actions: Actions with different results
                - skipped_actions: Actions that were skipped
                - match_rate: Percentage of matching actions
                - total_duration_ms: Replay duration in milliseconds
                - success: Whether replay completed successfully
                - action_results: Detailed results for each action
                - audit_mode: Whether audit mode was enabled

        Example:
            # Standard replay (re-executes search)
            result = mem.session_replay(session_id)

            # Audit replay with frozen context
            result = mem.session_replay(session_id, audit=True)

            # Audit replay with model comparison
            result = mem.session_replay(
                session_id,
                audit=True,
                use_model="groq:llama-3.3-70b-versatile",
                diff=True
            )
        """
        session_replay_fn = getattr(self._core, "session_replay", None)
        if session_replay_fn is None:
            raise RuntimeError("session_replay not available - replay feature not compiled")

        return session_replay_fn(
            session_id,
            top_k=top_k,
            adaptive=adaptive,
            audit=audit,
            use_model=use_model,
            diff=diff
        )

    def session_delete(self, session_id: str) -> bool:
        """Delete a recorded session.

        Args:
            session_id: UUID of the session to delete

        Returns:
            True if session was deleted, False if not found
        """
        session_delete_fn = getattr(self._core, "session_delete", None)
        if session_delete_fn is None:
            raise RuntimeError("session_delete not available - replay feature not compiled")
        return session_delete_fn(session_id)

    def session_checkpoint(self) -> Optional[str]:
        """Add a checkpoint to the current recording session.

        Checkpoints mark specific points in the session that can be
        used for partial replay or analysis.

        Returns:
            Checkpoint ID if in a session, None otherwise
        """
        session_checkpoint_fn = getattr(self._core, "session_checkpoint", None)
        if session_checkpoint_fn is None:
            raise RuntimeError("session_checkpoint not available - replay feature not compiled")
        return session_checkpoint_fn()

    # ─────────────────────────────────────────────────────────────────────────
    # Memory Cards & Enrichment
    # ─────────────────────────────────────────────────────────────────────────

    def memories(
        self,
        *,
        entity: Optional[str] = None,
        slot: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get memory cards (SPO triplets) stored in the memory.

        Memory cards represent structured knowledge extracted from frames
        in Subject-Predicate-Object (entity-slot-value) format.

        Args:
            entity: Optional entity name to filter by (e.g., "alice")
            slot: Optional slot/predicate to filter by (e.g., "employer")

        Returns:
            Dict with:
                - cards: List of memory card dicts
                - count: Total number of matching cards

        Example:
            >>> result = mem.memories(entity="alice")
            >>> for card in result["cards"]:
            ...     print(f"{card['entity']} -> {card['slot']}: {card['value']}")
        """
        memories_fn = getattr(self._core, "memories", None)
        if memories_fn is None:
            raise RuntimeError("memories not available - feature not compiled")
        return memories_fn(entity, slot)

    def memories_stats(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns counts of entities, cards, slots, and other memory-related metrics.

        Returns:
            Dict with:
                - entity_count: Number of unique entities
                - card_count: Total number of memory cards
                - slot_count: Number of unique slot types
                - cards_by_kind: Breakdown by card kind (fact, attribute, etc.)
                - enriched_frames: Number of frames that have been enriched
                - last_enrichment: Timestamp of last enrichment (or None)

        Example:
            >>> stats = mem.memories_stats()
            >>> print(f"Entities: {stats['entity_count']}, Cards: {stats['card_count']}")
        """
        memories_stats_fn = getattr(self._core, "memories_stats", None)
        if memories_stats_fn is None:
            raise RuntimeError("memories_stats not available - feature not compiled")
        return memories_stats_fn()

    def memory_entities(self) -> List[str]:
        """Get all entity names stored in memory.

        Returns:
            List of entity names (lowercase)

        Example:
            >>> entities = mem.memory_entities()
            >>> print(f"Known entities: {entities}")
        """
        memory_entities_fn = getattr(self._core, "memory_entities", None)
        if memory_entities_fn is None:
            raise RuntimeError("memory_entities not available - feature not compiled")
        return memory_entities_fn()

    def state(
        self,
        entity: str,
        *,
        slot: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get the current state of an entity (O(1) lookup).

        Returns the latest value for each slot of the entity.
        Uses SlotIndex for constant-time lookups.

        Args:
            entity: Entity name to query (e.g., "alice")
            slot: Optional specific slot to query (e.g., "employer")

        Returns:
            Dict with:
                - entity: The queried entity name
                - found: Whether the entity exists
                - slots: Dict mapping slot names to their current values

        Example:
            >>> state = mem.state("alice")
            >>> if state["found"]:
            ...     print(f"Alice's employer: {state['slots'].get('employer', {}).get('value')}")
        """
        state_fn = getattr(self._core, "state", None)
        if state_fn is None:
            raise RuntimeError("state not available - feature not compiled")
        return state_fn(entity, slot)

    def enrich(
        self,
        *,
        engine: str = "rules",
        force: bool = False,
    ) -> Dict[str, Any]:
        """Run enrichment to extract memory cards from frames.

        Enrichment scans frame text to extract structured knowledge as
        memory cards (SPO triplets).

        Args:
            engine: Engine to use - "rules" (default, fast pattern-based).
                    For LLM enrichment, use the CLI: memvid enrich --engine llm
            force: Re-enrich all frames, ignoring previous enrichment records

        Returns:
            Dict with:
                - engine: Engine name used
                - version: Engine version
                - frames_processed: Number of frames enriched
                - cards_extracted: Number of new cards extracted
                - total_cards: Total cards after enrichment
                - total_entities: Total entities after enrichment
                - new_cards: Net new cards added

        Example:
            >>> result = mem.enrich()
            >>> print(f"Extracted {result['cards_extracted']} cards from {result['frames_processed']} frames")
        """
        enrich_fn = getattr(self._core, "enrich", None)
        if enrich_fn is None:
            raise RuntimeError("enrich not available - feature not compiled")
        return enrich_fn(engine, force)

    def export_facts(
        self,
        *,
        format: str = "json",
        entity: Optional[str] = None,
        with_provenance: bool = False,
    ) -> str:
        """Export memory cards (facts/triplets) to various formats.

        Args:
            format: Output format - "json" (default), "csv", or "ntriples"
            entity: Optional entity filter
            with_provenance: Include source frame info (source_frame_id, timestamp, engine)

        Returns:
            String in the requested format

        Example:
            >>> # Export all facts as JSON
            >>> json_data = mem.export_facts()
            >>>
            >>> # Export Alice's facts as CSV with provenance
            >>> csv_data = mem.export_facts(format="csv", entity="alice", with_provenance=True)
            >>>
            >>> # Export as N-Triples for RDF tools
            >>> ntriples = mem.export_facts(format="ntriples")
        """
        export_facts_fn = getattr(self._core, "export_facts", None)
        if export_facts_fn is None:
            raise RuntimeError("export_facts not available - feature not compiled")
        return export_facts_fn(format, entity, with_provenance)

    def add_memory_cards(
        self,
        cards: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Add memory cards (SPO triplets) directly.

        This allows manual addition of extracted facts, useful when using
        external LLM enrichment or custom extraction logic.

        For automated LLM enrichment, use the CLI: memvid enrich --engine claude

        Args:
            cards: List of dicts with keys:
                - entity (required): Subject of the fact
                - slot (required): Predicate/relationship
                - value (required): Object/value
                - kind (optional): "Fact", "Preference", "Event", "Profile", "Relationship", "Other"
                - polarity (optional): "Positive", "Negative", "Neutral"
                - source_frame_id (optional): Frame this fact was extracted from
                - engine (optional): Extraction engine name (default: "sdk")
                - engine_version (optional): Engine version (default: "1.0.0")

        Returns:
            Dict with 'added' count and 'ids' list

        Example:
            >>> mem.add_memory_cards([
            ...     {"entity": "Alice", "slot": "employer", "value": "Acme Corp", "kind": "Fact"},
            ...     {"entity": "Alice", "slot": "role", "value": "Engineer", "kind": "Profile"},
            ...     {"entity": "Bob", "slot": "friend", "value": "Alice", "kind": "Relationship", "polarity": "Positive"},
            ... ])
            {'added': 3, 'ids': [1, 2, 3]}
        """
        add_memory_cards_fn = getattr(self._core, "add_memory_cards", None)
        if add_memory_cards_fn is None:
            raise RuntimeError("add_memory_cards not available - feature not compiled")
        return add_memory_cards_fn(cards)


def use(
    kind: Kind,
    filename: str,
    apikey: Optional[ApiKey] = None,
    *,
    mode: str = "open",
    enable_vec: bool = False,
    enable_lex: bool = True,
    read_only: bool = False,
    force_writable: bool = False,
    lock_timeout_ms: int = 250,
    force: Optional[str] = None,
    memvid_api_key: Optional[str] = None,
) -> Memvid:
    # Extract the actual API key string for _MemvidCore
    api_key_str = None
    if apikey is not None:
        if isinstance(apikey, str):
            api_key_str = apikey
        elif isinstance(apikey, dict):
            # If it's a dict, get the default key
            api_key_str = apikey.get("default")

    # Resolve memvid API key: param > apikey (if mv2_*) > global config > env var
    effective_memvid_key = memvid_api_key
    if not effective_memvid_key and api_key_str and api_key_str.startswith("mv2_"):
        effective_memvid_key = api_key_str
    if not effective_memvid_key:
        effective_memvid_key = _global_config.get("api_key") or os.environ.get("MEMVID_API_KEY")

    success = False
    try:
        core = _MemvidCore(
            filename,
            mode=mode,
            enable_lex=enable_lex,
            enable_vec=enable_vec,
            read_only=read_only,
            lock_timeout_ms=lock_timeout_ms,
            force=force,
            force_writable=force_writable,
            api_key=api_key_str,
        )
        normalized_apikey = _normalise_apikey(apikey)
        adapters = registry.resolve(str(kind), core, normalized_apikey)
        success = True
        return Memvid(kind=str(kind), core=core, attachments=adapters, memvid_api_key=effective_memvid_key)
    finally:
        is_create = mode == "create"
        is_open = mode == "open" and success
        track_command(filename, mode, success, is_create and success, is_open)


def create(
    filename: str,
    *,
    kind: Kind = "basic",
    apikey: Optional[ApiKey] = None,
    enable_vec: bool = False,
    enable_lex: bool = True,
    memory_id: Optional[str] = None,
    api_key: Optional[str] = None,
    memvid_api_key: Optional[str] = None,
) -> Memvid:
    """Create a new Memvid file and return a façade handle.

    If memory_id is provided (or set via configure()), automatically syncs
    tickets from the dashboard to apply plan capacity.

    Priority for memory_id:
    1. memory_id parameter (explicit)
    2. globalConfig.default_memory (set via configure())

    Named memories (set via configure(memories={"work": "abc123"})) are resolved
    automatically.

    Args:
        filename: Path to the mv2 file to create
        kind: Adapter kind (default: "basic")
        apikey: API keys for LLM providers
        enable_vec: Enable vector index (default: False)
        enable_lex: Enable lexical index (default: True)
        memory_id: Dashboard memory ID to bind to (auto-syncs tickets)
        api_key: Memvid API key (required if memory_id provided, or use MEMVID_API_KEY env var)

    Returns:
        Memvid instance with plan capacity if memory_id was provided
    """
    # Resolve the Memvid API key: explicit param > api_key param > global config > env var
    effective_api_key = (
        memvid_api_key
        or api_key
        or _global_config.get("api_key")
        or os.environ.get("MEMVID_API_KEY")
    )

    # Merge effective_api_key into apikey for the core
    effective_apikey = apikey
    if effective_api_key:
        if effective_apikey is None:
            effective_apikey = effective_api_key
        elif isinstance(effective_apikey, str):
            # Keep the LLM key as-is, effective_api_key will be passed separately
            pass

    mv = use(
        kind,
        filename,
        effective_api_key if effective_api_key else apikey,  # Pass memvid API key to core
        mode="create",
        enable_vec=enable_vec,
        enable_lex=enable_lex,
        memvid_api_key=effective_api_key,  # For query tracking
    )

    # Resolve memory_id: param > global config default
    effective_memory_id = memory_id or _global_config.get("default_memory")

    # Auto-sync tickets if memory_id is provided
    if effective_memory_id:
        # Resolve named memory (e.g., 'work' -> 'abc123')
        resolved_memory_id = resolve_memory(effective_memory_id)

        if not effective_api_key:
            raise ApiKeyRequiredError(
                "memory_id requires API key. Set via configure({'api_key': 'mv2_...'}), "
                "MEMVID_API_KEY env var, or pass api_key/memvid_api_key parameter. "
                "Get your API key at https://memvid.com/dashboard/api-keys"
            )
        # Convert 24-char MongoDB ObjectId to UUID format by padding with zeros
        normalized_id = resolved_memory_id.replace("-", "")
        if len(normalized_id) == 24 and all(c in "0123456789abcdefABCDEF" for c in normalized_id):
            normalized_id = normalized_id + "00000000"
        # Format as UUID with dashes
        if len(normalized_id) == 32:
            uuid_str = f"{normalized_id[:8]}-{normalized_id[8:12]}-{normalized_id[12:16]}-{normalized_id[16:20]}-{normalized_id[20:]}"
        else:
            uuid_str = resolved_memory_id  # Use as-is if already in UUID format

        # Resolve dashboard URL: global config > env var
        api_url = _global_config.get("dashboard_url") or os.environ.get("MEMVID_DASHBOARD_URL")
        mv.sync_tickets(uuid_str, effective_api_key, api_url)

    return mv


def lock(
    path: str,
    *,
    password: str,
    output: Optional[str] = None,
    force: bool = False,
) -> str:
    """Encrypt a `.mv2` file into an encrypted capsule (`.mv2e`)."""
    if _lock_capsule is None:
        raise RuntimeError("lock() support not available in this build")
    return str(_lock_capsule(path, password=password, output=output, force=force))


def unlock(
    path: str,
    *,
    password: str,
    output: Optional[str] = None,
    force: bool = False,
) -> str:
    """Decrypt a `.mv2e` capsule back into the original `.mv2` bytes."""
    if _unlock_capsule is None:
        raise RuntimeError("unlock() support not available in this build")
    return str(_unlock_capsule(path, password=password, output=output, force=force))


def lock_who(path: str) -> Dict[str, Any]:
    """Return lock status and owner information for a `.mv2` file."""
    if _lock_who is None:
        raise RuntimeError("lock_who support not available in this build")
    owner = _lock_who(path)
    return {"locked": owner is not None, "owner": owner}


def lock_nudge(path: str) -> bool:
    """Request a stale lock release for a `.mv2` file."""
    if _lock_nudge is None:
        raise RuntimeError("lock_nudge support not available in this build")
    return bool(_lock_nudge(path))


def verify_single_file(path: str) -> None:
    """Ensure no auxiliary files exist next to the `.mv2` (single-file guarantee)."""
    p = Path(path)
    parent = p.parent if p.parent != Path("") else Path(".")
    name = p.name
    offenders: List[str] = []
    for suffix in ("-wal", "-shm", "-lock", "-journal"):
        candidate = parent / f"{name}{suffix}"
        if candidate.exists():
            offenders.append(str(candidate))
    for suffix in (".wal", ".shm", ".lock", ".journal"):
        candidate = parent / f".{name}{suffix}"
        if candidate.exists():
            offenders.append(str(candidate))
    if not offenders:
        return
    err = CorruptFileError(
        "MV012: Auxiliary files detected next to the .mv2 (single-file guarantee violated)."
    )
    setattr(err, "code", "MV012")
    setattr(err, "offenders", offenders)
    raise err


def info() -> Dict[str, Any]:
    """Return SDK + native build information (for diagnostics and bug reports)."""
    try:
        from importlib import metadata as _metadata

        sdk_version = _metadata.version("memvid-sdk-free")
    except Exception:  # noqa: BLE001
        sdk_version = None

    native = _version_info() if callable(_version_info) else None
    return {
        "sdk_version": sdk_version,
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "native": native,
        "native_exports": [k for k in dir(_lib) if not k.startswith("_")],
    }


__all__ = [
    # Main class
    "Memvid",
    # Factory functions
    "use",
    "create",
    "lock",
    "unlock",
    # Global configuration
    "MemvidConfig",
    "configure",
    "get_config",
    "reset_config",
    "validate_config",
    "ConfigValidationResult",
    "ValidateConfigResult",
    "ApiKeyAclScope",
    "resolve_memory",
    "get_acl_scope_from_api_key",
    "acl_context_from_scope",
    "acl_metadata_from_scope",
    # Cloud memory management
    "create_memory",
    "list_memories",
    "CreateMemoryResult",
    "MemoryInfo",
    # Response types
    "StatsResult",
    "FindHit",
    "FindResult",
    "VecSearchResult",
    "AskResult",
    "AskStats",
    "AskUsage",
    "AskSource",
    "TimelineEntry",
    # Type aliases
    "Kind",
    # Base error
    "MemvidError",
    # Error helpers
    "get_error_suggestion",
    "format_error_with_suggestion",
    # Specific error classes (MV001-MV012)
    "CapacityExceededError",      # MV001
    "TicketInvalidError",         # MV002
    "TicketReplayError",          # MV003
    "LexIndexDisabledError",      # MV004
    "TimeIndexMissingError",      # MV005
    "VerifyFailedError",          # MV006
    "LockedError",                # MV007
    "ApiKeyRequiredError",        # MV008
    "MemoryAlreadyBoundError",    # MV009
    "FrameNotFoundError",         # MV010
    "VecIndexDisabledError",      # MV011
    "CorruptFileError",           # MV012
    "FileNotFoundError",          # MV013
    "VecDimensionMismatchError",  # MV014
    "EmbeddingFailedError",       # MV015
    "EncryptionError",            # MV016
    "NerModelNotAvailableError",  # MV017
    "ClipIndexDisabledError",     # MV018
    "QuotaExceededError",         # MV023
    # Introspection helpers
    "lock_who",
    "lock_nudge",
    "verify_single_file",
    "info",
    # Embeddings module
    "embeddings",
    # Analytics (opt-out: MEMVID_TELEMETRY=0)
    "flush_analytics",
    "is_telemetry_enabled",
]


def _warn_deprecated(name: str) -> None:
    warnings.warn(
        f"memvid_sdk.{name}() is deprecated; use memvid_sdk.use('basic', path) instead",
        DeprecationWarning,
        stacklevel=3,
    )


def open(*args, **kwargs):  # type: ignore[override]
    _warn_deprecated("open")
    return _open(*args, **kwargs)


def put(*args, **kwargs):  # type: ignore[override]
    _warn_deprecated("put")
    return _put(*args, **kwargs)


def find(*args, **kwargs):  # type: ignore[override]
    _warn_deprecated("find")
    return _find(*args, **kwargs)


def ask(*args, **kwargs):  # type: ignore[override]
    _warn_deprecated("ask")
    return _ask(*args, **kwargs)
