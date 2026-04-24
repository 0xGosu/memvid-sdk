"""
Analytics Module for Memvid Python SDK

Provides anonymous telemetry tracking with:
- Zero latency impact (fire-and-forget)
- In-memory batch with debounced flush
- SHA256-based anonymous IDs
- Opt-out via MEMVID_TELEMETRY=0
"""

from __future__ import annotations

import atexit
import hashlib
import os
import socket
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Configuration
ANALYTICS_ENDPOINT = "https://memvid.com/api/analytics/ingest"
FLUSH_INTERVAL_SECS = 5.0
MAX_BATCH_SIZE = 100
FLUSH_TIMEOUT_SECS = 5.0

# In-memory event queue
_event_queue: List[Dict[str, Any]] = []
_queue_lock = threading.Lock()
_flush_timer: Optional[threading.Timer] = None
_is_flushing = False

# Cached machine ID
_cached_machine_id: Optional[str] = None


def is_telemetry_enabled() -> bool:
    """Check if telemetry is enabled."""
    env = os.environ.get("MEMVID_TELEMETRY", "")
    return env != "0" and env.lower() != "false"


def _get_machine_id() -> str:
    """Get stable machine identifier (hashed for privacy)."""
    global _cached_machine_id
    if _cached_machine_id:
        return _cached_machine_id

    hasher = hashlib.sha256()

    # Add hostname
    try:
        hasher.update(socket.gethostname().encode())
    except Exception:
        pass

    # Add username
    try:
        hasher.update(os.getlogin().encode())
    except Exception:
        try:
            import getpass
            hasher.update(getpass.getuser().encode())
        except Exception:
            pass

    # Add home directory
    try:
        hasher.update(os.path.expanduser("~").encode())
    except Exception:
        pass

    hasher.update(b"memvid_telemetry_python_v1")
    _cached_machine_id = hasher.hexdigest()[:16]
    return _cached_machine_id


def generate_anon_id(file_path: Optional[str] = None) -> str:
    """
    Generate anonymous ID for tracking.
    Format: anon_<16 hex chars> or paid_<16 hex chars>
    """
    # Check for API key (paid user)
    api_key = os.environ.get("MEMVID_API_KEY", "")
    if api_key and len(api_key) >= 8:
        prefix = api_key[:8]
        hasher = hashlib.sha256()
        hasher.update(prefix.encode())
        hasher.update(b"memvid_paid_v1")
        return f"paid_{hasher.hexdigest()[:16]}"

    # Free user - use machine ID
    machine_id = _get_machine_id()
    hasher = hashlib.sha256()
    hasher.update(machine_id.encode())
    if file_path:
        hasher.update(file_path.encode())
    hasher.update(b"memvid_anon_v1")
    return f"anon_{hasher.hexdigest()[:16]}"


def generate_file_hash(file_path: str) -> str:
    """Generate hash for file path (privacy-preserving)."""
    hasher = hashlib.sha256()
    # Normalize path
    try:
        normalized = os.path.abspath(file_path)
    except Exception:
        normalized = file_path
    hasher.update(normalized.encode())
    hasher.update(b"memvid_file_v1")
    return hasher.hexdigest()[:16]


def _schedule_flush() -> None:
    """Schedule a flush if not already scheduled."""
    global _flush_timer
    if _flush_timer is None and _event_queue:
        _flush_timer = threading.Timer(FLUSH_INTERVAL_SECS, _do_flush)
        _flush_timer.daemon = True
        _flush_timer.start()


def track_event(event: Dict[str, Any]) -> None:
    """Queue an analytics event."""
    if not is_telemetry_enabled():
        return

    with _queue_lock:
        _event_queue.append(event)
        _schedule_flush()

        # Force flush if queue is getting large
        if len(_event_queue) >= MAX_BATCH_SIZE:
            global _flush_timer
            if _flush_timer:
                _flush_timer.cancel()
                _flush_timer = None
            # Flush in background thread
            threading.Thread(target=_do_flush, daemon=True).start()


def track_command(
    file_path: Optional[str],
    command: str,
    success: bool,
    file_created: bool = False,
    file_opened: bool = False,
) -> None:
    """Convenience function to track a command."""
    if not is_telemetry_enabled():
        return

    event = {
        "anon_id": generate_anon_id(file_path),
        "file_hash": generate_file_hash(file_path) if file_path else "none",
        "client": "python",
        "command": command,
        "success": success,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "file_created": file_created,
        "file_opened": file_opened,
    }
    track_event(event)


def _get_ssl_context():
    """Get SSL context with proper certificate handling for macOS."""
    import ssl

    # Try certifi first (most reliable)
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass

    # Try macOS certificate installer path
    try:
        import sys
        if sys.platform == "darwin":
            # macOS Python installations often need this
            import subprocess
            # Check if certificates are installed
            ctx = ssl.create_default_context()
            return ctx
    except Exception:
        pass

    # Fallback: create unverified context (last resort for analytics only)
    # This is acceptable for analytics since we're not sending sensitive data
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    except Exception:
        return None


def _do_flush() -> None:
    """Flush events to server."""
    global _flush_timer, _is_flushing

    with _queue_lock:
        _flush_timer = None
        if _is_flushing or not _event_queue:
            return
        _is_flushing = True
        events = list(_event_queue)
        _event_queue.clear()

    try:
        endpoint = os.environ.get("MEMVID_ANALYTICS_URL", ANALYTICS_ENDPOINT)

        # Use urllib to avoid external dependencies
        import json
        from urllib.request import Request, urlopen
        from urllib.error import URLError

        payload = json.dumps({"events": events}).encode("utf-8")
        req = Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        # Get SSL context for macOS compatibility
        ssl_context = _get_ssl_context()

        try:
            with urlopen(req, timeout=FLUSH_TIMEOUT_SECS, context=ssl_context):
                pass  # Success
        except URLError:
            # Re-queue events on failure (up to max batch size)
            with _queue_lock:
                remaining_capacity = MAX_BATCH_SIZE - len(_event_queue)
                if remaining_capacity > 0:
                    _event_queue[:0] = events[:remaining_capacity]
    except Exception:
        # Silently ignore all errors
        pass
    finally:
        with _queue_lock:
            _is_flushing = False


def flush() -> None:
    """Force flush all pending events (call before process exit)."""
    global _flush_timer
    with _queue_lock:
        if _flush_timer:
            _flush_timer.cancel()
            _flush_timer = None
    _do_flush()


# Flush on process exit
atexit.register(flush)
