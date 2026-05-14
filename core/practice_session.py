"""Manage real-time practice session state without audio hardware.

Pure Python only — no sounddevice, no UI, no pitch detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.live_feedback import ready_targets


@dataclass
class PracticeSession:
    """Stateful wrapper around ``ready_targets`` that tracks evaluation history.

    Parameters
    ----------
    targets          : ordered list of target dicts with a ``"time"`` key in beats.
    bpm              : tempo in beats per minute.
    count_in_beats   : count-in length in beats.
    sample_rate      : audio sample rate in Hz.
    evaluated_indices: set of target indices already evaluated (mutated in place).
    """

    targets:           list[dict]
    bpm:               float
    count_in_beats:    int
    sample_rate:       int
    evaluated_indices: set[int] = field(default_factory=set)

    def update(self, current_sample: int) -> list[int]:
        """Return newly ready target indices and mark them as evaluated.

        Each index is returned at most once across all calls.

        Parameters
        ----------
        current_sample : current position in the audio buffer (samples).

        Returns
        -------
        list[int]
            Indices of targets that just became ready, in ascending order.
            Empty if no new targets are ready.
        """
        newly_ready = ready_targets(
            self.targets,
            self.bpm,
            self.count_in_beats,
            self.sample_rate,
            current_sample,
            self.evaluated_indices,
        )
        self.evaluated_indices.update(newly_ready)
        return newly_ready
