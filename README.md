# 🧠 **NeuroVoice — Preprocesamiento de Datos**

## 🎯 **Objetivo del Proyecto**
El propósito de esta etapa del proyecto **NeuroVoice** es preparar un conjunto de audios de voz balanceado y estandarizado, listo para el entrenamiento de un modelo que analice patrones de voz asociados a la enfermedad de **Parkinson**.  

El pipeline abarca los siguientes pasos principales:  

1️⃣ Dataset balanceado (40 sanos/40 enfermos)
2️⃣ Normalización de frecuencia a **16 kHz**  
3️⃣ Duración estandarizada a **3.0 segundos**  
4️⃣ Extracción de **features** (MFCC y Mel-spectrogramas)  
5️⃣ **Aumento de datos** y **balance final 100/100** tanto para audios sanos como para audios enfermos

---

## ⚙️ **Requisitos del Entorno**

**Versión recomendada:** Python ≥ 3.10  

### 🧩 Crear entorno virtual (recomendado)

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

💡 En Visual Studio Code, selecciona el intérprete:

```Python: Select Interpreter → .venv\Scripts\python.exe```


##  **📦 Archivo de dependencias — requirements.txt**

Para instalar dependencias: 

```pip install -r requirements.txt```




## **🚀 Pasos del Pipeline**
### 1️⃣ Inventario del Dataset (RAW → Metadata)

Genera el archivo metadata_master.csv y un reporte de duraciones.

```
python .\preprocesamiento\01_make_metadata.py `
  --root data\raw `
  --outmeta data\metadata\metadata_master.csv `
  --reportdir data\metadata\reports
```

### 2️⃣ Normalización (16kHz) + Duración (3s)

Procesa todos los audios y genera versiones uniformes en data/processed/....

```
python .\preprocesamiento\02_resample_fix_duration.py `
  --inmeta data\metadata\metadata_master.csv `
  --outmeta data\metadata\metadata_processed.csv `
  --outroot data\processed `
  --sr 16000 `
  --duration 3.0 `
  --vad_db 25
```

### 3️⃣ Extracción de Features (MFCC + Mel)

Convierte los audios en representaciones numéricas .npy y guarda la información en metadata_features.csv.

```
python .\preprocesamiento\03_extract_features.py `
  --inmeta data\metadata\metadata_processed.csv `
  --outmeta data\metadata\metadata_features.csv `
  --outdir data\features
```

### 4️⃣ Visualización de Features (opcional)

Permite visualizar las diferencias entre una voz sana y una afectada por Parkinson mediante Mel-spectrogramas y MFCC.

```python .\preprocesamiento\04_visualize_features.py --audio NOMBRE_DEL_AUDIO.wav```


### 5️⃣ Balance Final (100/100 por clase)

Copia la base procesada y genera únicamente los aumentos necesarios hasta alcanzar 100 audios por clase

```
python .\preprocesamiento\05_balance_to_target.py `
  --inmeta data\metadata\metadata_processed.csv `
  --outmeta data\metadata\metadata_final_for_training.csv `
  --finalroot data\final_for_training `
  --target_per_class 100 `
  --sr 16000 `
  --duration 3.0 `
  --seed 42
```

#### 📊 Resultados Esperados

| Carpeta                    | Contenido                   | Descripción                        |
| -------------------------- | --------------------------- | ---------------------------------- |
| `data/processed/`          | Audios normalizados         | 16kHz / 3s, limpios y uniformes    |
| `data/features/`           | Matrices `.npy`             | MFCC y Mel-spectrogramas           |
| `data/final_for_training/` | Audios listos para entrenar | 100 sanos + 100 enfermos           |
| `data/metadata/`           | CSVs con registros          | Trazabilidad completa del pipeline |


## **🧠 Interpretación de los Features**

### ***MFCC (20 × 94): resumen de la envolvente espectral de la voz.***
- Filas → coeficientes cepstrales (timbre)
- Columnas → tiempo

### **Mel-Spectrograma (128 × 94): energía por banda de frecuencia.**
- Eje Y → frecuencia (escala Mel)
- Eje X → tiempo
- Color → intensidad sonora

Una voz sana presenta patrones suaves, definidos y estables.
Una voz con Parkinson suele mostrar irregularidades, energía reducida y variaciones abruptas