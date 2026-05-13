"""Fixed-size ring buffer for storing recent audio samples.

Tracks absolute sample position so callers can request windows by
global sample index rather than buffer-relative offsets.

No sounddevice. No audio hardware. Pure Python / NumPy only.
"""

from __future__ import annotations

import numpy as np


class RingBuffer:
    """Fixed-capacity circular buffer keyed by absolute sample index.

    Parameters
    ----------
    sample_rate : int | float
        Audio sample rate in Hz.
    max_seconds : float
        How many seconds of audio to retain.
    channels : int
        Number of audio channels (1 = mono).
    """

    def __init__(self, sample_rate: float, max_seconds: float, channels: int = 1) -> None:
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")
        self._capacity = int(sample_rate * max_seconds)
        self._channels = channels
        self._buf = np.zeros((self._capacity, channels), dtype=np.float64)
        self._total_written = 0

    # ── public API ─────────────────────────────────────────────────────────────

    def add(self, chunk: np.ndarray) -> None:
        """Write *chunk* into the buffer, overwriting oldest samples if full.

        Parameters
        ----------
        chunk : np.ndarray
            Shape ``(n,)`` for mono or ``(n, channels)`` for multi-channel.
            A mono buffer also accepts ``(n, 1)`` shaped input.
        """
        data = self._normalise(chunk)
        n = data.shape[0]
        if n == 0:
            return

        self._total_written += n

        # Only the most recent _capacity samples fit; discard older excess.
        if n > self._capacity:
            data = data[-self._capacity:]
            n = self._capacity

        write_start = (self._total_written - n) % self._capacity

        end = write_start + n
        if end <= self._capacity:
            self._buf[write_start:end] = data
        else:
            # Wraparound: split into two writes.
            first = self._capacity - write_start
            self._buf[write_start:] = data[:first]
            self._buf[: n - first] = data[first:]

    def current_sample(self) -> int:
        """Absolute index of the next sample to be written (exclusive end)."""
        return self._total_written

    def available_range(self) -> tuple[int, int]:
        """``(oldest_available, current_sample)`` as absolute sample indices.

        The range ``[oldest_available, current_sample)`` is currently held.
        Before the buffer has filled, oldest_available is 0.
        """
        oldest = max(0, self._total_written - self._capacity)
        return (oldest, self._total_written)

    def get_window(self, start_sample: int, end_sample: int) -> np.ndarray:
        """Return samples in ``[start_sample, end_sample)`` as an array.

        Returns
        -------
        np.ndarray
            Shape ``(n,)`` for mono buffers, ``(n, channels)`` for multi-channel.

        Raises
        ------
        ValueError
            If the requested range is not fully within ``available_range()``.
        """
        n = end_sample - start_sample
        if n <= 0:
            raise ValueError(
                f"start_sample ({start_sample}) must be < end_sample ({end_sample})"
            )

        oldest, current = self.available_range()
        if start_sample < oldest:
            raise ValueError(
                f"start_sample {start_sample} is too old; oldest available is {oldest}"
            )
        if end_sample > current:
            raise ValueError(
                f"end_sample {end_sample} is beyond current_sample {current}"
            )

        start_pos = start_sample % self._capacity
        end_pos = start_pos + n

        if end_pos <= self._capacity:
            result = self._buf[start_pos:end_pos].copy()
        else:
            first = self._capacity - start_pos
            result = np.concatenate([self._buf[start_pos:], self._buf[: n - first]])

        if self._channels == 1:
            return result.squeeze(axis=1)
        return result

    # ── internal ───────────────────────────────────────────────────────────────

    def _normalise(self, chunk: np.ndarray) -> np.ndarray:
        """Return chunk reshaped to ``(n, channels)``."""
        arr = np.asarray(chunk, dtype=np.float64)
        if arr.ndim == 1:
            if self._channels != 1:
                raise ValueError(
                    f"1-D chunk supplied to a {self._channels}-channel buffer"
                )
            return arr.reshape(-1, 1)
        if arr.ndim == 2:
            if arr.shape[1] != self._channels:
                raise ValueError(
                    f"chunk has {arr.shape[1]} channel(s); buffer expects {self._channels}"
                )
            return arr
        raise ValueError(f"chunk must be 1-D or 2-D, got shape {arr.shape}")
