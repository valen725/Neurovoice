import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly

class Recorder:
    def __init__(self, samplerate: int = 44100, channels: int = 1):
        self.samplerate = samplerate
        self.channels = channels

    def record(self, seconds: int = 3) -> np.ndarray:
        audio = sd.rec(int(seconds * self.samplerate), samplerate=self.samplerate, channels=self.channels, dtype='float32')
        sd.wait()
        audio = np.squeeze(audio)
        return audio

class Player:
    def play(self, audio: np.ndarray, samplerate: int):
        sd.play(audio, samplerate=samplerate, blocking=False)

def float_to_int16(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)

def save_wav_16k(path: str, audio: np.ndarray, sr: int):
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    # Avoid division by zero
    max_val = np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else 1.0
    audio = audio / max_val
    # Resample if needed
    if sr != 16000:
        from math import gcd
        g = gcd(sr, 16000)
        up = 16000 // g
        down = sr // g
        audio = resample_poly(audio, up, down)
        sr = 16000
    pcm16 = float_to_int16(audio)
    sf.write(path, pcm16, sr, subtype='PCM_16')