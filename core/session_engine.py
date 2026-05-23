"""Event-driven session coordinator for onset-based timing feedback.

Pure Python — no sounddevice, no audio buffer, no hardware dependencies.

This module provides a push/onset-event path that complements the pull/buffer
path in ``core/live_pipeline``.  Both paths coexist; neither replaces the other.

Typical usage
-------------
    engine = SessionEngine(targets, bpm=120.0, count_in_beats=2)

    # called once per detected onset (from audio callback or test):
    events = engine.on_onset(onset_time_s)

    # called once per audio block or sim tick to emit missed-target events:
    events = engine.update_time(current_time_s)

Both methods return a (possibly empty) list of feedback event dicts whose
fields are compatible with those emitted by ``core.feedback_events.feedback_event``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.feedback_events import feedback_event
from core.target_windows import target_audio_time_s
from core.tempo_tracker import TempoTracker
from core.timing_policy import match_window_s as _match_window_s


@dataclass
class SessionEngine:
    """Onset-event-driven session coordinator.

    Parameters
    ----------
    targets
        Ordered list of target dicts, each with a ``"time"`` key in beats.
    bpm
        Nominal tempo in beats per minute.
    count_in_beats
        Count-in length in beats before the first target.
    tracker
        Optional :class:`~core.tempo_tracker.TempoTracker`.  When provided,
        ``observe()`` is called for every onset that matches a target.
    match_window_s
        Half-width of the onset-acceptance window in seconds.  An onset is
        considered a candidate for target *i* if
        ``|onset_time_s - nominal_i| <= match_window_s``.
        Defaults to half a beat (``30.0 / bpm``).
    evaluated_indices
        Set of target indices already scored.  Shared or pre-seeded externally
        if needed; normally left at the default empty set.
    """

    targets:           list[dict]
    bpm:               float
    count_in_beats:    int
    tracker:           TempoTracker | None = None
    match_window_s:    float | None = None
    evaluated_indices: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.match_window_s is None:
            self.match_window_s = _match_window_s(self.bpm)

    # ── Public API ─────────────────────────────────────────────────────────────

    def on_onset(self, onset_time_s: float) -> list[dict]:
        """Route one detected onset to a pending target.

        Parameters
        ----------
        onset_time_s
            Absolute time of the detected onset in seconds.

        Returns
        -------
        list[dict]
            A single-element list containing a feedback event dict if the onset
            matched a pending target, otherwise an empty list.

        Notes
        -----
        Matching rule: the pending target whose reference time is closest to
        *onset_time_s* and within ``match_window_s`` is selected.  Ties
        resolve to the lower target index.

        When a tracker is present and anchored, the reference time is
        ``tracker.adjusted_target_time(nominal_i)`` — the tracker's prediction
        of where the player's beat actually falls.  Before the anchor is set
        (first beat) or when no tracker is provided, the reference falls back
        to the nominal grid time.  This lets the match window follow the
        player's actual tempo under large drift while keeping ``timing_error_s``
        always relative to the nominal grid (musical feedback).
        """
        best_idx   = None
        best_delta = float("inf")

        use_adjusted = self.tracker is not None and self.tracker.has_anchor

        for i in range(len(self.targets)):
            if i in self.evaluated_indices:
                continue
            nom = self._nominal_s(i)
            ref = self.tracker.adjusted_target_time(nom) if use_adjusted else nom
            delta = abs(onset_time_s - ref)
            if delta <= self.match_window_s and delta < best_delta:
                best_delta = delta
                best_idx   = i

        if best_idx is None:
            return []

        nom            = self._nominal_s(best_idx)
        timing_error_s = onset_time_s - nom

        if self.tracker is not None:
            self.tracker.observe(nom, onset_time_s)

        self.evaluated_indices.add(best_idx)

        result = {
            "detected_note":     "?",
            "timing_error_s":    timing_error_s,
            "pitch_error_cents": None,
            "confidence":        None,
        }
        ev = feedback_event(best_idx, self.targets[best_idx], result)
        ev["onset_time_s"] = onset_time_s
        return [ev]

    def update_time(self, current_time_s: float) -> list[dict]:
        """Emit miss events for targets whose acceptance windows have closed.

        A target is considered missed when
        ``current_time_s > nominal_time_s + match_window_s`` and it has not
        already been evaluated (hit or previously missed).

        Parameters
        ----------
        current_time_s
            Current session time in seconds.

        Returns
        -------
        list[dict]
            Miss event dicts in ascending target-index order.  Empty if no
            new targets have been missed.
        """
        events: list[dict] = []

        for i in range(len(self.targets)):
            if i in self.evaluated_indices:
                continue
            deadline = self._nominal_s(i) + self.match_window_s
            if current_time_s > deadline:
                self.evaluated_indices.add(i)
                result = {
                    "detected_note":     None,
                    "timing_error_s":    None,
                    "pitch_error_cents": None,
                    "confidence":        None,
                }
                events.append(feedback_event(i, self.targets[i], result))

        return events

    # ── Private helpers ────────────────────────────────────────────────────────

    def _nominal_s(self, idx: int) -> float:
        """Absolute nominal time (seconds) for target *idx*."""
        return target_audio_time_s(self.targets, idx, self.bpm, self.count_in_beats)
