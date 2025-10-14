import sounddevice as sd
from scipy.io.wavfile import write

fs = 16000  # Frecuencia de muestreo
seconds = 3  # Duración de la grabación

import os
folder = "data/raw/HC"
os.makedirs(folder, exist_ok=True)
# Contar archivos existentes
existing = len([f for f in os.listdir(folder) if f.endswith(".wav")])
next_num = existing + 1
filename = f"{folder}/{next_num}.wav"

print(" Habla ahora (mantén la vocal 'a')")
audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()
write(filename, fs, audio)
print(f"Grabación guardada como {filename}")

