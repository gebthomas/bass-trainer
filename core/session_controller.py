"""Phase-aware orchestration layer for a real-time practice session.

Bridges SessionEngine (onset matching / miss detection) and SessionLog
(telemetry) behind a simple state-machine interface.  No audio hardware
dependency.

Typical usage
-------------
    controller = SessionController(targets, bpm=80.0, count_in_beats=2,
                                   sample_rate=44100)
    controller.start()

    # once per audio block:
    events = controller.update(current_sample, onset_times_s=[...])
    for ev in events:
        print(ev["severity"], ev["messages"])

    if controller.is_complete():
        print(controller.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from core.log_metrics import LogMetrics, compute_log_metrics
from core.session_engine import SessionEngine
from core.session_log import (
    SCHEMA_VERSION,
    TARGET_HIT,
    TARGET_MISS,
    SessionEvent,
    SessionLog,
    append_event,
)


class SessionPhase(Enum):
    WAITING  = "waiting"
    COUNT_IN = "count_in"
    ACTIVE   = "active"
    COMPLETE = "complete"
    ABORTED  = "aborted"


_TERMINAL = frozenset({SessionPhase.COMPLETE, SessionPhase.ABORTED})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionController:
    """Phase-aware controller for a single practice session.

    Parameters
    ----------
    targets
        Ordered target dicts, each with a ``"time"`` key (beats after count-in).
    bpm
        Nominal tempo in beats per minute.
    count_in_beats
        Number of count-in beats before the first target.
    sample_rate
        Audio sample rate in Hz; used to convert ``current_sample`` → seconds.
    """

    targets:        list[dict]
    bpm:            float
    count_in_beats: int
    sample_rate:    int

    _phase:  SessionPhase             = field(default=SessionPhase.WAITING, init=False, repr=False)
    _engine: Optional[SessionEngine]  = field(default=None,                 init=False, repr=False)
    _log:    Optional[SessionLog]     = field(default=None,                 init=False, repr=False)

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def phase(self) -> SessionPhase:
        return self._phase

    @property
    def log(self) -> Optional[SessionLog]:
        """The session telemetry log, or ``None`` before ``start()`` is called."""
        return self._log

    def start(self) -> None:
        """Initialise engine and log; transition WAITING → COUNT_IN.

        Raises
        ------
        RuntimeError
            If called in any phase other than WAITING.
        """
        if self._phase != SessionPhase.WAITING:
            raise RuntimeError(
                f"start() called in phase {self._phase.value!r}; "
                "only valid from WAITING"
            )
        self._engine = SessionEngine(self.targets, self.bpm, self.count_in_beats)
        self._log    = SessionLog(schema_version=SCHEMA_VERSION, started_at=_now_iso())
        self._phase  = SessionPhase.COUNT_IN

    def update(
        self,
        current_sample: int,
        onset_times_s: list[float] | None = None,
    ) -> list[dict]:
        """Advance session state and return new feedback events.

        Parameters
        ----------
        current_sample
            Sample position since ``start()`` was called.
        onset_times_s
            Onset times in seconds from session start detected since the last
            ``update()`` call.  ``None`` or empty → no new onsets this tick.

        Returns
        -------
        list[dict]
            Feedback event dicts (empty when the phase is not ACTIVE or no
            targets changed state this tick).  Each dict contains at minimum:
            ``target_idx``, ``severity``, ``messages``, ``timing_error_s``.
        """
        if self._phase in _TERMINAL or self._phase == SessionPhase.WAITING:
            return []

        current_time_s = current_sample / self.sample_rate

        # COUNT_IN → ACTIVE once the count-in window has elapsed
        if self._phase == SessionPhase.COUNT_IN:
            count_in_s = self.count_in_beats * 60.0 / self.bpm
            if current_time_s >= count_in_s:
                self._phase = SessionPhase.ACTIVE

        if self._phase != SessionPhase.ACTIVE:
            return []

        new_events: list[dict] = []

        for onset_s in (onset_times_s or []):
            new_events.extend(self._engine.on_onset(onset_s))

        new_events.extend(self._engine.update_time(current_time_s))

        for ev in new_events:
            self._record(ev, current_time_s)

        if len(self._engine.evaluated_indices) == len(self.targets):
            self._finalize(SessionPhase.COMPLETE)

        return new_events

    def abort(self) -> None:
        """Transition to ABORTED; no-op if already in a terminal phase."""
        if self._phase in _TERMINAL:
            return
        self._finalize(SessionPhase.ABORTED)

    def is_complete(self) -> bool:
        """Return True when all targets have been evaluated (phase == COMPLETE)."""
        return self._phase == SessionPhase.COMPLETE

    def summary(self) -> Optional[LogMetrics]:
        """Return aggregate metrics once the session has ended.

        Returns ``None`` while the session is still in progress or if
        ``start()`` has not been called.
        """
        if self._log is None or self._phase not in _TERMINAL:
            return None
        return compute_log_metrics(self._log)

    # ── Private helpers ─────────────────────────────────────────────────────

    def _record(self, ev: dict, current_time_s: float) -> None:
        """Append a feedback-event dict to the session log."""
        is_hit     = ev.get("detected_note") is not None
        time_sec   = ev.get("onset_time_s", current_time_s)
        event_type = TARGET_HIT if is_hit else TARGET_MISS
        value      = ev.get("timing_error_s") if is_hit else None

        append_event(
            self._log,
            SessionEvent(
                time_sec     = max(0.0, float(time_sec)),
                event_type   = event_type,
                target_index = ev.get("target_idx"),
                value        = value,
                message      = ", ".join(ev.get("messages", [])) or None,
            ),
        )

    def _finalize(self, phase: SessionPhase) -> None:
        self._phase = phase
        if self._log is not None:
            self._log.ended_at = _now_iso()
