import matplotlib
matplotlib.use('Agg')  # Configurar backend no-interactivo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa.display
from pathlib import Path
import os

# Cargar metadata
df = pd.read_csv("data/metadata/metadata_features.csv")

# Selecciona un ejemplo sano y uno enfermo
sano = df[df["label"] == "healthy"].iloc[0]
enfermo = df[df["label"] == "parkinson"].iloc[0]

# Carga features
mel_sano = np.load(sano["mel_path"])
mel_enf = np.load(enfermo["mel_path"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Configurar el primer subplot (SANO)
im1 = librosa.display.specshow(mel_sano, x_axis="time", y_axis="mel", sr=16000, 
                               cmap="magma", ax=ax1)
ax1.set_title(f"SANO\n{sano['filename'][:30]}...", fontsize=12, pad=20)
ax1.set_xlabel("Tiempo (s)", fontsize=10)
ax1.set_ylabel("Frecuencia Mel", fontsize=10)

# Colorbar para el primer subplot
cbar1 = plt.colorbar(im1, ax=ax1, format="%+2.0f dB", shrink=0.8)
cbar1.set_label("Amplitud (dB)", fontsize=10)

# Configurar el segundo subplot (ENFERMO) 
im2 = librosa.display.specshow(mel_enf, x_axis="time", y_axis="mel", sr=16000, 
                               cmap="magma", ax=ax2)
ax2.set_title(f"PARKINSON\n{enfermo['filename'][:30]}...", fontsize=12, pad=20)
ax2.set_xlabel("Tiempo (s)", fontsize=10)
ax2.set_ylabel("Frecuencia Mel", fontsize=10)

# Colorbar para el segundo subplot
cbar2 = plt.colorbar(im2, ax=ax2, format="%+2.0f dB", shrink=0.8)
cbar2.set_label("Amplitud (dB)", fontsize=10)

# Título general
fig.suptitle("Comparación de Mel-Spectrogramas ", fontsize=16, y=0.95)

# Ajustar espaciado para evitar solapamientos
plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15, wspace=0.4)

# Crear directorio para visualizaciones
os.makedirs("data/visualizations", exist_ok=True)

# Guardar la imagen con mejor calidad
output_path = "data/visualizations/mel_spectrograms_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
            edgecolor='none', pad_inches=0.2)

print(f" Visualización guardada en: {output_path}")


plt.close()  # Liberar memoria
