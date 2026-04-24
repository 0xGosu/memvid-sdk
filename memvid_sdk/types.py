"""Typed result objects returned by memvid-sdk-free bindings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(slots=True)
class Hit:
    frame_id: int
    uri: str
    title: Optional[str] = None
    rank: int = 0
    score: float = 0.0
    matches: int = 0
    snippet: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    track: Optional[str] = None
    created_at: Optional[str] = None
    content_dates: List[str] = field(default_factory=list)


@dataclass(slots=True)
class FindResult:
    query: str
    hits: List[Hit]
    took_ms: int
    total_hits: int
    engine: str
    context: str
    next_cursor: Optional[str] = None


@dataclass(slots=True)
class Source:
    uri: str
    frame_id: int
    title: Optional[str]
    score: float
    snippet: Optional[str]


@dataclass(slots=True)
class FollowUp:
    """Follow-up suggestions when answer has low confidence."""
    needed: bool
    reason: str
    hint: str
    available_topics: List[str]
    suggestions: List[str]


@dataclass(slots=True)
class AskResult:
    question: str
    answer: Optional[str]
    mode: str
    retriever: str
    context_only: bool
    context: str
    hits: List[Hit]
    sources: List[Source]
    stats: Dict[str, int]
    usage: Dict[str, int]
    follow_up: Optional[FollowUp] = None


@dataclass(slots=True)
class TimelineEntry:
    frame_id: int
    timestamp: int
    preview: str
    uri: Optional[str]
    child_frames: List[int]


__all__ = [
    "Hit",
    "FindResult",
    "Source",
    "FollowUp",
    "AskResult",
    "TimelineEntry",
]
