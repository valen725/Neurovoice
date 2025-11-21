import numpy as np
import librosa

def preprocess_wav_for_model(audio, sr, target_sr=16000, duration=3.0, denoise=False):
    """
    Preprocesa un array de audio para el modelo NeuroVoice (idéntico a predict_fixed.py):
      - Convierte a mono
      - Resamplea a target_sr
      - Aplica VAD simple (ventana de máxima energía)
      - Normaliza
      - Recorta o padd a duración fija (centrado)
    """
    # Convertir a mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Resamplear
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    # VAD simple (igual que en predict_fixed.py)
    window_size = len(audio) // 10
    energies = []
    for i in range(0, len(audio) - window_size, window_size // 2):
        window = audio[i:i + window_size]
        energy = np.sum(window ** 2)
        energies.append(energy)
    if energies:
        max_energy_idx = np.argmax(energies)
        start_sample = max_energy_idx * (window_size // 2)
        expand_samples = int(target_sr * 2)
        start = max(0, start_sample - expand_samples // 2)
        end = min(len(audio), start_sample + expand_samples // 2)
        audio = audio[start:end] if end > start else audio
    # Normalizar
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.95
    # Recortar o pad a duración fija (centrado)
    target_len = int(target_sr * duration)
    if len(audio) > target_len:
        start = (len(audio) - target_len) // 2
        audio = audio[start:start + target_len]
    elif len(audio) < target_len:
        pad_left = (target_len - len(audio)) // 2
        pad_right = target_len - len(audio) - pad_left
        audio = np.pad(audio, (pad_left, pad_right), mode='constant')
    return audio, target_sr

def mel_spectrogram_db(audio, sr, n_mels=128, n_fft=1024, hop_length=512):
    """
    Calcula el mel-spectrograma en dB para graficar.
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db