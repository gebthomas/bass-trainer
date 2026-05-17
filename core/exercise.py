"""Shared exercise/session configuration model.

Pure dataclasses and stdlib only — no audio hardware, no file I/O.

An Exercise bundles everything needed to configure a practice session:
tempo, targets, onset-detection sensitivity, and tracker tuning. It can
be serialised to/from a plain dict (for JSON round-tripping by callers)
and converts itself into constructor kwargs for each engine class.

Typical usage
-------------
    import json
    from core.exercise import Exercise
    from core.onset_adapter import OnsetAdapter
    from core.session_engine import SessionEngine
    from core.tempo_tracker import TempoTracker

    data    = json.loads(path.read_text())
    ex      = Exercise.from_dict(data)

    tracker = TempoTracker(ex.bpm, **ex.to_tracker_kwargs())
    engine  = SessionEngine(**ex.to_session_engine_kwargs(), tracker=tracker)
    adapter = OnsetAdapter(sample_rate=sample_rate, **ex.to_onset_adapter_kwargs())

Note: ``sample_rate`` is intentionally absent from Exercise — it is a
hardware property supplied at runtime, not an exercise property.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


# ── Sub-configs ───────────────────────────────────────────────────────────────

@dataclass
class OnsetAdapterConfig:
    """Onset-detection thresholds for OnsetAdapter.

    Defaults match the OnsetAdapter class defaults exactly so that omitting
    this config from a session dict produces identical behaviour to the demo.
    """

    min_rms:      float = 0.018
    min_peak:     float = 0.15
    refractory_s: float = 0.150

    def __post_init__(self) -> None:
        if self.min_rms < 0:
            raise ValueError(f"min_rms must be >= 0, got {self.min_rms}")
        if self.min_peak < 0:
            raise ValueError(f"min_peak must be >= 0, got {self.min_peak}")
        if self.refractory_s <= 0:
            raise ValueError(f"refractory_s must be > 0, got {self.refractory_s}")


@dataclass
class TrackerConfig:
    """Tuning parameters for TempoTracker.

    Defaults match the TempoTracker.__init__ defaults exactly.
    """

    phase_alpha:           float = 0.10
    tempo_beta:            float = 0.30
    outlier_threshold:     float = 0.40
    confidence_window:     int   = 8
    drift_window:          int   = 4
    drift_threshold_scale: float = 2.0
    drift_min_frac:        float = 0.10


# ── Exercise ──────────────────────────────────────────────────────────────────

@dataclass
class Exercise:
    """Shared configuration model for a bass practice session.

    Parameters
    ----------
    name
        Human-readable exercise name.
    bpm
        Nominal tempo in beats per minute.
    count_in_beats
        Number of count-in beats before the first target. These are played
        (or simulated) before the session clock starts; targets are timed
        relative to the first beat after the count-in.
    targets
        Ordered list of target dicts.  Each must have a ``"time"`` key
        (positive number, beats) and may have a ``"note"`` key (str).
        Times must be strictly increasing.
    match_window_s
        Half-width of the onset-acceptance window in seconds.
        ``None`` means SessionEngine computes its default (half a beat).
    onset
        Onset-detection thresholds.  Defaults to OnsetAdapterConfig defaults.
    tracker
        Tempo/phase tracker tuning.  Defaults to TrackerConfig defaults.
    """

    name:           str
    bpm:            float
    count_in_beats: int
    targets:        list[dict]
    match_window_s: float | None          = None
    onset:          OnsetAdapterConfig    = field(default_factory=OnsetAdapterConfig)
    tracker:        TrackerConfig         = field(default_factory=TrackerConfig)

    def __post_init__(self) -> None:
        if self.bpm <= 0:
            raise ValueError(f"bpm must be > 0, got {self.bpm}")
        if self.count_in_beats < 0:
            raise ValueError(f"count_in_beats must be >= 0, got {self.count_in_beats}")
        if len(self.targets) < 1:
            raise ValueError("targets must not be empty")
        for i, t in enumerate(self.targets):
            if "time" not in t:
                raise ValueError(f"target[{i}] is missing required 'time' key")
            if not isinstance(t["time"], (int, float)) or t["time"] <= 0:
                raise ValueError(
                    f"target[{i}]['time'] must be a positive number, got {t['time']!r}"
                )
        times = [t["time"] for t in self.targets]
        for i in range(1, len(times)):
            if times[i] <= times[i - 1]:
                raise ValueError(
                    f"target times must be strictly increasing; "
                    f"targets[{i}]['time']={times[i]} <= targets[{i - 1}]['time']={times[i - 1]}"
                )
        if self.match_window_s is not None and self.match_window_s <= 0:
            raise ValueError(f"match_window_s must be > 0, got {self.match_window_s}")

    # ── Serialisation ──────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict) -> "Exercise":
        """Construct an Exercise from a plain dict (e.g. parsed from JSON).

        Sub-dicts for ``onset`` and ``tracker`` are optional and may be sparse —
        missing keys fall back to class defaults.
        """
        onset_data   = data.get("onset",   {})
        tracker_data = data.get("tracker", {})
        return cls(
            name           = data["name"],
            bpm            = float(data["bpm"]),
            count_in_beats = int(data["count_in_beats"]),
            targets        = list(data["targets"]),
            match_window_s = data.get("match_window_s"),
            onset          = OnsetAdapterConfig(**onset_data),
            tracker        = TrackerConfig(**tracker_data),
        )

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict representation.

        ``match_window_s`` is omitted when ``None`` so that reloading the dict
        via ``from_dict`` preserves the "engine computes default" behaviour.
        """
        d = dataclasses.asdict(self)
        if d.get("match_window_s") is None:
            d.pop("match_window_s", None)
        return d

    # ── Conversion to constructor kwargs ───────────────────────────────────────

    def to_session_engine_kwargs(self) -> dict:
        """Return kwargs for ``SessionEngine()``.

        Does not include ``tracker`` or ``evaluated_indices`` — the caller
        constructs the tracker separately and passes it in.
        """
        kwargs: dict = {
            "targets":        self.targets,
            "bpm":            self.bpm,
            "count_in_beats": self.count_in_beats,
        }
        if self.match_window_s is not None:
            kwargs["match_window_s"] = self.match_window_s
        return kwargs

    def to_onset_adapter_kwargs(self) -> dict:
        """Return kwargs for ``OnsetAdapter()``.

        Does not include ``sample_rate`` — that is a hardware property
        supplied by the caller at runtime.
        """
        return {
            "min_rms":      self.onset.min_rms,
            "min_peak":     self.onset.min_peak,
            "refractory_s": self.onset.refractory_s,
        }

    def to_tracker_kwargs(self) -> dict:
        """Return kwargs for ``TempoTracker()``.

        Does not include ``nominal_bpm`` — pass ``ex.bpm`` as the first
        positional argument.
        """
        return dataclasses.asdict(self.tracker)
