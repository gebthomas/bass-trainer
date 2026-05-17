"""Adaptive real-time tempo/phase tracker for music practice feedback.

Pure state class only — no audio hardware, no sounddevice.

Timing model
------------
The tracker maintains two degrees of freedom:

  tempo_ratio   : adjusted_beat_duration / nominal_beat_duration
                  > 1 → player is slower than nominal; < 1 → faster
  phase_offset  : residual constant offset (seconds) after tempo correction
                  positive → player is consistently late relative to the grid

Given those, the adjusted target time for any nominal beat position is::

    adjusted(nominal_t) = anchor_actual
                        + (nominal_t - anchor_nominal) * tempo_ratio
                        + phase_offset

where anchor_nominal / anchor_actual are the nominal / actual times of the
first accepted observation.  Using an explicit anchor avoids entangling the
count-in duration with the per-beat tempo ratio.

Update strategy
---------------
Each call to ``observe()`` computes the prediction error against the current
adjusted grid.  Errors beyond ``outlier_threshold * nominal_beat_s`` are
treated as isolated mistakes and silently ignored (the grid is not moved).
For accepted observations:

  * ``phase_offset`` is nudged by ``phase_alpha * error``  (slow EMA)
  * ``tempo_ratio`` is nudged toward the measured inter-onset ratio
    ``delta_actual / delta_nominal`` with rate ``tempo_beta``.  The default
    rate has been raised from 0.05 to 0.15 so that 48 beats of sustained drift
    produce near-full convergence (residual ≈ 0.05 % vs ≈ 9 % at the old rate)
    while still being slow enough that isolated timing mistakes do not retrain
    the grid.

Both learning rates are intentionally small so that sustained drift is tracked
without reacting to individual timing mistakes.

Confidence
----------
``confidence()`` is based on the mean squared residual over a sliding window
of recent accepted observations, normalised so that a quarter-beat RMS maps to
≈ 0.5.  It rises to 1.0 as timing becomes perfectly consistent and falls
toward 0 for erratic playing.
"""

from __future__ import annotations

from collections import deque


class TempoTracker:
    """Adaptive tracker for player tempo and phase relative to a nominal grid.

    Parameters
    ----------
    nominal_bpm         : Reference tempo in BPM (e.g. the metronome setting).
    phase_alpha         : EMA learning rate for phase offset (0 < α ≤ 1).
    tempo_beta          : EMA learning rate for tempo ratio (0 < β ≤ 1).
    outlier_threshold   : Errors larger than this fraction of a beat are
                          treated as isolated mistakes and ignored.
    confidence_window   : Sliding window size for the confidence metric.
    tempo_ratio_bounds  : (min, max) clamp for the tempo ratio to prevent
                          runaway adaptation.
    """

    def __init__(
        self,
        nominal_bpm: float,
        phase_alpha: float = 0.10,
        tempo_beta: float = 0.30,
        outlier_threshold: float = 0.40,
        confidence_window: int = 8,
        tempo_ratio_bounds: tuple[float, float] = (0.5, 2.0),
    ) -> None:
        if nominal_bpm <= 0:
            raise ValueError(f"nominal_bpm must be positive, got {nominal_bpm}")
        self._nominal_beat_s = 60.0 / nominal_bpm
        self._tempo_ratio = 1.0
        self._phase_offset = 0.0

        self._phase_alpha = phase_alpha
        self._tempo_beta = tempo_beta
        self._outlier_limit_s = outlier_threshold * self._nominal_beat_s
        self._tempo_min, self._tempo_max = tempo_ratio_bounds

        # Anchor is set lazily from the first accepted observation
        self._anchor_nominal: float | None = None
        self._anchor_actual: float | None = None
        # Last accepted (nominal, actual) pair, for inter-onset tempo measurement
        self._prev_obs: tuple[float, float] | None = None

        self._recent_residuals: deque[float] = deque(maxlen=confidence_window)

    # ── Public API ────────────────────────────────────────────────────────────

    def observe(self, nominal_beat_time: float, actual_onset_time: float) -> None:
        """Record a successful onset and update tempo/phase estimates.

        Parameters
        ----------
        nominal_beat_time : When the metronome grid says this beat occurs (s).
        actual_onset_time : When the player actually played it (s).
        """
        if self._anchor_nominal is None:
            # First observation: set anchor and record as prev; no update yet.
            self._anchor_nominal = nominal_beat_time
            self._anchor_actual = actual_onset_time
            self._prev_obs = (nominal_beat_time, actual_onset_time)
            return

        predicted = self.adjusted_target_time(nominal_beat_time)
        error = actual_onset_time - predicted

        if abs(error) > self._outlier_limit_s:
            # Isolated mistake — don't retrain the grid.
            return

        # Gradually shift the phase estimate.
        self._phase_offset += self._phase_alpha * error

        # Gradually update tempo from inter-onset intervals.
        if self._prev_obs is not None:
            prev_nom, prev_act = self._prev_obs
            nom_gap = nominal_beat_time - prev_nom
            if nom_gap > 1e-6:
                act_gap = actual_onset_time - prev_act
                measured_ratio = act_gap / nom_gap
                self._tempo_ratio += self._tempo_beta * (measured_ratio - self._tempo_ratio)
                self._tempo_ratio = max(self._tempo_min, min(self._tempo_max, self._tempo_ratio))

        self._prev_obs = (nominal_beat_time, actual_onset_time)

        # Track residual after update for confidence estimation.
        residual = actual_onset_time - self.adjusted_target_time(nominal_beat_time)
        self._recent_residuals.append(residual)

    def adjusted_target_time(self, nominal_beat_time: float) -> float:
        """Return the estimated actual time for a beat at *nominal_beat_time*.

        Before the first observation this is the identity (returns the nominal
        time unchanged).
        """
        if self._anchor_nominal is None:
            return nominal_beat_time
        elapsed_nominal = nominal_beat_time - self._anchor_nominal
        return self._anchor_actual + elapsed_nominal * self._tempo_ratio + self._phase_offset

    def current_tempo_bpm(self) -> float:
        """Return the tracker's current estimated player tempo in BPM."""
        adjusted_beat_s = self._nominal_beat_s * self._tempo_ratio
        return 60.0 / adjusted_beat_s

    def confidence(self) -> float:
        """Return a 0-1 stability score.

        0 = no data or highly erratic; 1 = perfectly consistent.
        A quarter-beat RMS maps to ≈ 0.5.
        """
        n = len(self._recent_residuals)
        if n < 2:
            return 0.0
        mean_sq = sum(r * r for r in self._recent_residuals) / n
        scale = (0.25 * self._nominal_beat_s) ** 2
        return 1.0 / (1.0 + mean_sq / scale)

    # ── Diagnostic properties (read-only) ────────────────────────────────────

    @property
    def tempo_ratio(self) -> float:
        """Adjusted beat duration relative to nominal (1.0 = on tempo)."""
        return self._tempo_ratio

    @property
    def phase_offset(self) -> float:
        """Residual phase offset in seconds (positive = player runs late)."""
        return self._phase_offset

    @property
    def has_anchor(self) -> bool:
        """True once the first observation has been recorded."""
        return self._anchor_nominal is not None

    @property
    def outlier_limit_s(self) -> float:
        """Absolute error (seconds) beyond which observations are treated as outliers."""
        return self._outlier_limit_s

    @property
    def nominal_beat_s(self) -> float:
        """Nominal beat duration in seconds (60 / nominal_bpm)."""
        return self._nominal_beat_s
