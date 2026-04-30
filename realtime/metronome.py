import time
import threading
import numpy as np


def play_click(frequency=1000, duration=0.04, volume=0.4, sample_rate=48000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    click = volume * np.sin(2 * np.pi * frequency * t)
    return click


class Metronome:
    def __init__(self, bpm, mode="count_in_and_click", count_in_beats=4,
                 volume=0.35, beats_per_measure=4, sample_rate=48000):
        self.bpm = bpm
        self.mode = mode
        self.count_in_beats = count_in_beats
        self.volume = volume
        self.beats_per_measure = beats_per_measure
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.play_start_time = None
        self.sample_rate = sample_rate

    def start(self, play_start_time):
        self.play_start_time = play_start_time
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def _run(self):
        if self.mode == "silent":
            return

        beat_duration = 60.0 / self.bpm
        count_in_duration = self.count_in_beats * beat_duration
        next_click_time = self.play_start_time - count_in_duration
        beat = 1
        click_phase = "count_in" if self.count_in_beats > 0 else "play"

        if self.count_in_beats == 0:
            print(f"\nPlay start at {self.play_start_time:.3f}s")
            print("PLAY\n")

        while not self.stop_event.is_set():
            now = time.perf_counter()

            if now >= next_click_time:
                if click_phase == "count_in":
                    if beat == 1:
                        print(f"\nCount-in at {self.bpm} BPM")
                    freq = 1200 if beat == 1 else 800
                    click = play_click(frequency=freq, duration=0.035, volume=self.volume, sample_rate=self.sample_rate)
                    import sounddevice as sd
                    sd.play(click, self.sample_rate, blocking=False)

                    beat += 1
                    if beat > self.count_in_beats:
                        if self.mode == "count_in_and_click":
                            click_phase = "play"
                            beat = 1
                            print("PLAY\n")
                        else:
                            return
                else:
                    freq = 1200 if beat == 1 else 800
                    click = play_click(frequency=freq, duration=0.035, volume=self.volume, sample_rate=self.sample_rate)
                    import sounddevice as sd
                    sd.play(click, self.sample_rate, blocking=False)
                    beat += 1
                    if beat > self.beats_per_measure:
                        beat = 1

                next_click_time += beat_duration
            else:
                time.sleep(0.001)
