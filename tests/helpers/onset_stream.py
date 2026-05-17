"""Simulated onset-stream harness for SessionEngine testing.

Provides a deterministic, hardware-free way to drive SessionEngine
through a sequence of onset events spread over simulated time.

This module can be swapped for a real-clock or audio-replay driver
later without changing the tests that call it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.session_engine import SessionEngine


def simulate_onset_stream(
    engine: SessionEngine,
    onsets: list[float],
    session_end_s: float,
    tick_s: float = 0.020,
) -> list[dict]:
    """Advance simulated time and drive *engine* through an onset stream.

    Time advances from 0 to *session_end_s* in discrete steps of *tick_s*
    (index-multiplied to avoid cumulative float drift).  At each step:

      1. All pending onsets whose time <= current tick time are injected via
         ``engine.on_onset()``.  Onsets are processed before the miss check
         so an onset arriving exactly at the deadline is treated as a hit.
      2. ``engine.update_time(current_time_s)`` is called to emit miss events
         for any target whose acceptance window has closed.

    This mirrors the structure of a real audio callback loop: onsets arrive
    as they are detected, and the miss-detection cursor advances in step with
    simulated wall time.

    Parameters
    ----------
    engine        : ``SessionEngine`` instance (mutated in place).
    onsets        : Onset times in seconds.  Need not be pre-sorted.
    session_end_s : Simulated time at which the loop terminates.
    tick_s        : Tick interval in seconds (default 20 ms ≈ one audio block
                    at 48 kHz / 1024 frames).

    Returns
    -------
    list[dict]
        All events emitted by the engine, in the order they were generated.
        Each dict has the fields produced by ``feedback_event()`` plus the
        ``onset_time_s`` field added by ``SessionEngine.on_onset()``.
    """
    events: list[dict] = []
    sorted_onsets = sorted(onsets)
    onset_idx = 0
    tick_count = 0

    while True:
        t = tick_count * tick_s
        if t > session_end_s + 1e-9:
            break

        # Inject onsets that have arrived by this tick (onset before miss check).
        while onset_idx < len(sorted_onsets) and sorted_onsets[onset_idx] <= t:
            events.extend(engine.on_onset(sorted_onsets[onset_idx]))
            onset_idx += 1

        # Advance the miss-detection cursor.
        events.extend(engine.update_time(t))

        tick_count += 1

    return events
