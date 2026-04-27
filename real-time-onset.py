import sounddevice as sd
import numpy as np
import time
from collections import deque

DEVICE_ID = 1
SAMPLE_RATE = 48000
CHANNELS = 2
BLOCK_SIZE = 1024

CHANNEL_INDEX = 0      # channel 1 on MiniMe
MIN_RMS = 0.015
RISE_RATIO = 1.7
REFRACTORY_MS = 90
HISTORY_BLOCKS = 4

start_time = None
last_onset_time = -999
energy_history = deque(maxlen=HISTORY_BLOCKS)


def callback(indata, frames, callback_time, status):
    global start_time, last_onset_time

    if status:
        print(status)

    if start_time is None:
        start_time = time.perf_counter()

    audio = indata[:, CHANNEL_INDEX]

    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))

    recent_energy = np.mean(energy_history) if energy_history else rms
    energy_history.append(rms)

    now = time.perf_counter()
    elapsed = now - start_time
    time_since_last = (elapsed - last_onset_time) * 1000

    strong_enough = rms > MIN_RMS
    rising_fast = rms > recent_energy * RISE_RATIO
    outside_refractory = time_since_last > REFRACTORY_MS

    if strong_enough and rising_fast and outside_refractory:
        last_onset_time = elapsed
        print(
            f"Onset at {elapsed:8.3f} s | "
            f"peak={peak:.3f} | rms={rms:.3f} | "
            f"baseline={recent_energy:.3f}"
        )


print("Listening for note onsets. Press Ctrl+C to stop.")

with sd.InputStream(
    device=DEVICE_ID,
    channels=CHANNELS,
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    dtype="float32",
    callback=callback,
):
    while True:
        time.sleep(0.1)