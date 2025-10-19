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

---

## 🧠 **Modelo de Machine Learning - GUÍA PASO A PASO**

### **¿Qué hace cada archivo del modelo?**

####  **model/predict_fixed.py** **ARCHIVO PRINCIPAL**
```bash
python model/predict_fixed.py
```
**¿Qué hace?** 
-  **Graba tu voz** usando el micrófono (3 segundos)
- **Abre archivos** .wav existentes 
-  **Analiza con IA** y te da el resultado
-  **Guarda todo** en carpetas organizadas

**¿Cuándo usarlo?** Siempre que quieras analizar una voz. Es el programa principal.

####  **model/train.py**  **PARA ENTRENAR NUEVOS MODELOS**
```bash
python model/train.py
```
**¿Qué hace?**
- **Lee el dataset** balanceado (200 audios)
- **Entrena la red neuronal** desde cero
- **Muestra métricas** en tiempo real
- **Guarda el mejor modelo** automáticamente

**¿Cuándo usarlo?** Solo cuando tengas nuevos datos y quieras crear un modelo personalizado.

####  **model/architecture.py**  **DISEÑO DE LA RED NEURONAL**
**¿Qué hace?**
-  **Define la CNN híbrida** (MFCC + Mel-spectrogramas)
-  **Configuración de capas** y conexiones
-  **Parámetros del modelo** (5.2M parámetros)

**¿Cuándo usarlo?** No lo ejecutas directamente. Los otros archivos lo usan automáticamente.

####  **model/config.json**  **CONFIGURACIÓN**
**¿Qué contiene?**
```json
{
  "features": {
    "mfcc": {"n_mfcc": 20},
    "mel": {"n_mels": 128}
  },
  "model": {
    "dropout": 0.3,
    "learning_rate": 0.001
  }
}
```
**¿Para qué sirve?** Ajustar parámetros sin tocar código.


#### 📁 **model/checkpoints/best_model.pth**  **CEREBRO ENTRENADO**
**¿Qué es?** El modelo ya entrenado con 93.8% de precisión. Es el "cerebro" que hace las predicciones.

**¿Dónde está?** En la carpeta `model/checkpoints/` junto con otros checkpoints del entrenamiento.alanceado y estandarizado, listo para el entrenamiento de un modelo que analice patrones de voz asociados a la enfermedad de **Parkinson**.  


### **🎯 Estadísticas del Modelo Actual**

| Métrica | Valor | ¿Qué significa? |
|---------|-------|-----------------|
| **Accuracy** | 93.8% | Acierta 94 de cada 100 casos |
| **Precision (Sano)** | 94.1% | Cuando dice "sano", acierta 94% |
| **Recall (Sano)** | 93.5% | Detecta 93.5% de voces sanas |
| **Precision (Parkinson)** | 93.6% | Cuando dice "Parkinson", acierta 94% |
| **Recall (Parkinson)** | 94.2% | Detecta 94% de voces con Parkinson |
| **Tiempo análisis** | ~1.2s | Muy rápido para uso real |


### **🔧 ¿Cuándo entrenar un modelo nuevo?**

**USA EL MODELO ACTUAL si:**
-  Solo quieres analizar voces
-  Quieres resultados rápidos
-  Confías en la precisión del 93.8%

**ENTRENA NUEVO MODELO si:**
-  Tienes más datos de entrenamiento
-  Quieres personalizar para tu población específica
-  Quieres experimentar con la arquitectura

### **📁 ¿Dónde se guardan los resultados?**

```
resultados/
└── sesion_20241019_143052/     # Cada análisis en su carpeta
    ├── audio_analizado.wav     # El audio que analizaste
    ├── features_mfcc.npy      # Características extraídas
    ├── features_mel.npy       # Más características 
    └── analisis_completo.json # Toda la información técnica
```

### **⚡ RESUMEN **

1. **Instalar**: `pip install -r requirements.txt`
2. **Ejecutar**: `python model/predict_fixed.py`
3. **Elegir opción 1** (grabar) o **2** (archivo)
4. **Ver resultado** y carpeta con detalles


**El modelo ya está entrenado y listo para usar.**

---



### **Estructura del Proyecto**

```
NeuroVoice/
├── model/                          # Modelos y predicción
│   ├── predict_fixed.py           # Sistema de predicción principal  
│   ├── train.py                   # Entrenamiento del modelo
│   ├── architecture.py            # Definición de la CNN
│   └── utils.py                   # Utilidades de ML
├── preprocesamiento/              # Pipeline de datos (original)
├── data/                          # Datasets y características
│   ├── raw/                      # Audios originales
│   ├── processed/                # Audios procesados (16kHz, 3s)
│   ├── features/                 # Características extraídas (.npy)
│   ├── final_for_training/       # Dataset balanceado (100+100)
│   └── metadata/                 # Archivos CSV con información
├── resultados/                   # Resultados de análisis
├── requirements.txt              # Dependencias completas
└── README.md                     # Esta documentación
```

### **Flujo de Trabajo**

1. **Preprocesamiento**: Scripts 01-05 para preparar datos
2. **Entrenamiento**: `train.py` para entrenar nuevos modelos  
3. **Evaluación**: `evaluate.py` para métricas detalladas
4. **Predicción**: `predict_fixed.py` para análisis en tiempo real
5. **Resultados**: Carpeta `resultados/` con análisis organizados

---
