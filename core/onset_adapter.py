"""Thin adapter that converts raw audio blocks to session-time onset events.

Pure Python + NumPy — no sounddevice, no threads, no hardware dependencies.

Typical usage
-------------
    adapter = OnsetAdapter(sample_rate=48000)
    session = SessionEngine(targets, bpm=120.0, count_in_beats=2)

    # called once per audio block (from sounddevice callback or read loop):
    onset_times = adapter.process_block(block_start_sample, audio_block)
    for t in onset_times:
        session.on_onset(t)
    session.update_time(block_start_sample / adapter.sample_rate)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OnsetAdapter:
    """Convert mono audio blocks to session-time onset event times.

    Parameters
    ----------
    sample_rate
        Audio sample rate in Hz (e.g. 44100, 48000).
    min_rms
        Minimum block RMS required to trigger an onset via the sustained-sound
        path.  Catches notes whose energy is spread across the block.
    min_peak
        Minimum block peak amplitude required to trigger an onset via the
        transient path.  Catches sharp attacks (plucked bass, slap) whose
        energy is concentrated in a few samples and would fall below min_rms.
        Detection fires when ``rms >= min_rms OR peak >= min_peak``.
    refractory_s
        Minimum gap in seconds between successive onsets.  A block that would
        produce an onset sooner than this after the previous one is suppressed.
        Note: at 16th-note density above ~100 BPM the inter-note gap (~150 ms)
        approaches this window; lower refractory_s if supporting dense patterns.
    """

    sample_rate:        int
    min_rms:            float = 0.018
    min_peak:           float = 0.15
    refractory_s:       float = 0.150
    _last_onset_sample: int | None = field(default=None, init=False, repr=False)

    def process_block(
        self,
        block_start_sample: int,
        audio_block: np.ndarray,
    ) -> list[float]:
        """Detect at most one onset in *audio_block* and return its session time.

        Parameters
        ----------
        block_start_sample
            Absolute sample index of the first sample in *audio_block*.
            Sample 0 corresponds to the moment the audio stream opened (i.e.
            the start of the count-in).
        audio_block
            Mono float32/float64 audio samples, shape ``(N,)``.

        Returns
        -------
        list[float]
            A single-element list ``[onset_time_s]`` if an onset was detected,
            otherwise an empty list.  The onset time is
            ``onset_sample / sample_rate`` where
            ``onset_sample = block_start_sample + argmax(|audio_block|)``.
        """
        rms  = float(np.sqrt(np.mean(audio_block ** 2)))
        peak = float(np.max(np.abs(audio_block)))
        if rms < self.min_rms and peak < self.min_peak:
            return []

        if self._last_onset_sample is not None:
            refractory_samples = int(self.refractory_s * self.sample_rate)
            if block_start_sample - self._last_onset_sample < refractory_samples:
                return []

        onset_sample = block_start_sample + int(np.argmax(np.abs(audio_block)))
        self._last_onset_sample = onset_sample
        return [onset_sample / self.sample_rate]
