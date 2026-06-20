"""Data structures for take comparison: onset records, match records, and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MatchCategory(Enum):
    MATCHED = "matched"
    A_ONLY = "a_only"
    B_ONLY = "b_only"
    AMBIGUOUS = "ambiguous"
    NOISE = "noise"


@dataclass
class OnsetRecord:
    """A single detected onset from one take, with extensible feature slots."""
    time_s: float
    strength: float
    amplitude_db: float
    raw_time_s: float
    take_label: str
    onset_index: int

    pitch_hz: Optional[float] = None
    pitch_note: Optional[str] = None
    pitch_confidence: Optional[float] = None


@dataclass
class MatchRecord:
    """A single comparison result between two takes' onsets."""
    category: MatchCategory
    onset_a: Optional[OnsetRecord] = None
    onset_b: Optional[OnsetRecord] = None

    timing_diff_ms: Optional[float] = None
    amplitude_diff_db: Optional[float] = None
    pitch_diff_cents: Optional[float] = None

    candidates_b: list[OnsetRecord] = field(default_factory=list)
    confidence: float = 1.0
    notes: str = ""

    @property
    def time_s(self) -> float:
        if self.onset_a is not None:
            return self.onset_a.time_s
        if self.onset_b is not None:
            return self.onset_b.time_s
        return 0.0


@dataclass
class ComparisonResult:
    """Full result of comparing two takes."""
    take_a_path: str
    take_b_path: str
    alignment_offset_s: float
    alignment_confidence: float
    sample_rate: int
    matches: list[MatchRecord] = field(default_factory=list)

    @property
    def matched_count(self) -> int:
        return sum(1 for m in self.matches if m.category == MatchCategory.MATCHED)

    @property
    def a_only_count(self) -> int:
        return sum(1 for m in self.matches if m.category == MatchCategory.A_ONLY)

    @property
    def b_only_count(self) -> int:
        return sum(1 for m in self.matches if m.category == MatchCategory.B_ONLY)

    @property
    def ambiguous_count(self) -> int:
        return sum(1 for m in self.matches if m.category == MatchCategory.AMBIGUOUS)

    @property
    def noise_count(self) -> int:
        return sum(1 for m in self.matches if m.category == MatchCategory.NOISE)

    @property
    def timing_diffs_ms(self) -> list[float]:
        return [m.timing_diff_ms for m in self.matches
                if m.category == MatchCategory.MATCHED and m.timing_diff_ms is not None]

    @property
    def amplitude_diffs_db(self) -> list[float]:
        return [m.amplitude_diff_db for m in self.matches
                if m.category == MatchCategory.MATCHED and m.amplitude_diff_db is not None]
