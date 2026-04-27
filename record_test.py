import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

device_id = 1
sample_rate = 48000
duration = 5

audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype="float32",
    device=device_id,
    mapping=[1],
    blocking=True
)

print("Peak:", np.max(np.abs(audio)))
print("RMS:", np.sqrt(np.mean(audio**2)))

write("bass_ch1_test.wav", sample_rate, audio)
print("Saved bass_ch1_test.wav")