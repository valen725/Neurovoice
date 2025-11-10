# **NeuroVoice — Sistema de Detección de Parkinson por Análisis de Voz**

testeando pr

## **Objetivo del Proyecto**

El propósito de esta etapa del proyecto **NeuroVoice** es preparar un conjunto de audios de voz balanceado y estandarizado, listo para el entrenamiento de un modelo que analice patrones de voz asociados a la enfermedad de **Parkinson**.

El pipeline abarca los siguientes pasos principales:

1. Dataset balanceado (40 sanos/40 enfermos)
2. Normalización de frecuencia a **16 kHz**
3. Duración estandarizada a **3.0 segundos**
4. Extracción de **features** (MFCC y Mel-spectrogramas)
5. **Aumento de datos** y **balance final 100/100** tanto para audios sanos como para audios enfermos

---

## **Requisitos del Entorno**

**Versión recomendada:** Python ≥ 3.10

### Crear entorno virtual (recomendado)

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

En Visual Studio Code, selecciona el intérprete:

`Python: Select Interpreter → .venv\Scripts\python.exe`

## **Archivo de dependencias — requirements.txt**

Para instalar dependencias:

`pip install -r requirements.txt`

## **Pasos del Pipeline**

### 1. Inventario del Dataset (RAW → Metadata)

Genera el archivo metadata_master.csv y un reporte de duraciones.

```
python .\preprocesamiento\01_make_metadata.py `
  --root data\raw `
  --outmeta data\metadata\metadata_master.csv `
  --reportdir data\metadata\reports
```

### 2. Normalización (16kHz) + Duración (3s)

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

### 3. Extracción de Features (MFCC + Mel)

Convierte los audios en representaciones numéricas .npy y guarda la información en metadata_features.csv.

```
python .\preprocesamiento\03_extract_features.py `
  --inmeta data\metadata\metadata_processed.csv `
  --outmeta data\metadata\metadata_features.csv `
  --outdir data\features
```

### 4. Visualización de Features (opcional)

Permite visualizar las diferencias entre una voz sana y una afectada por Parkinson mediante Mel-spectrogramas y MFCC.

`python .\preprocesamiento\04_visualize_features.py --audio NOMBRE_DEL_AUDIO.wav`

### 5. Balance Final (100/100 por clase)

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

### 6. Corrección de Metadata (opcional)

**¿Cuándo usar?** Solo si hay inconsistencias entre el metadata y los archivos reales.

```bash
python preprocesamiento/06_fix_metadata.py
```

**¿Qué hace?**

- Escanea automáticamente `data/final_for_training/`
- Crea metadata actualizado con todos los archivos encontrados
- Hace backup del metadata anterior
- Sincroniza archivos reales con el inventario

#### Resultados Esperados

| Carpeta                    | Contenido                   | Descripción                        |
| -------------------------- | --------------------------- | ---------------------------------- |
| `data/processed/`          | Audios normalizados         | 16kHz / 3s, limpios y uniformes    |
| `data/features/`           | Matrices `.npy`             | MFCC y Mel-spectrogramas           |
| `data/final_for_training/` | Audios listos para entrenar | 100 sanos + 100 enfermos           |
| `data/metadata/`           | CSVs con registros          | Trazabilidad completa del pipeline |

## **Interpretación de los Features**

### **_MFCC (20 × 94): resumen de la envolvente espectral de la voz._**

- Filas → coeficientes cepstrales (timbre)
- Columnas → tiempo

### **Mel-Spectrograma (128 × 94): energía por banda de frecuencia.**

- Eje Y → frecuencia (escala Mel)
- Eje X → tiempo
- Color → intensidad sonora

Una voz sana presenta patrones suaves, definidos y estables.
Una voz con Parkinson suele mostrar irregularidades, energía reducida y variaciones abruptas

---

## **Modelo de Machine Learning - GUÍA PASO A PASO**

### **¿Qué hace cada archivo del modelo?**

#### **model/predict_fixed.py** **ARCHIVO PRINCIPAL**

```bash
python model/predict_fixed.py
```

**¿Qué hace?**

- **Graba tu voz** usando el micrófono (3 segundos)
- **Abre archivos** .wav existentes
- **Analiza con IA** y te da el resultado
- **Guarda todo** en carpetas organizadas

**¿Cuándo usarlo?** Siempre que quieras analizar una voz. Es el programa principal.

#### **model/train.py** **PARA ENTRENAR NUEVOS MODELOS**

```bash
python model/train.py
```

**¿Qué hace?**

- **Lee el dataset** balanceado (200 audios)
- **Entrena la red neuronal** desde cero
- **Muestra métricas** en tiempo real
- **Guarda el mejor modelo** automáticamente

**¿Cuándo usarlo?** Solo cuando tengas nuevos datos y quieras crear un modelo personalizado.

#### **model/architecture.py** **DISEÑO DE LA RED NEURONAL**

**¿Qué hace?**

- **Define la CNN híbrida** (MFCC + Mel-spectrogramas)
- **Configuración de capas** y conexiones
- **Parámetros del modelo** (5.2M parámetros)

**¿Cuándo usarlo?** No lo ejecutas directamente. Los otros archivos lo usan automáticamente.

#### **model/config.json** **CONFIGURACIÓN**

**¿Qué contiene?**

```json
{
  "features": {
    "mfcc": { "n_mfcc": 20 },
    "mel": { "n_mels": 128 }
  },
  "model": {
    "dropout": 0.3,
    "learning_rate": 0.001
  }
}
```

**¿Para qué sirve?** Ajustar parámetros sin tocar código.

#### **model/checkpoints/best_model.pth** **CEREBRO ENTRENADO**

**¿Qué es?** El modelo ya entrenado con 93.8% de precisión. Es el "cerebro" que hace las predicciones.

**¿Dónde está?** En la carpeta `model/checkpoints/` junto con otros checkpoints del entrenamiento.alanceado y estandarizado, listo para el entrenamiento de un modelo que analice patrones de voz asociados a la enfermedad de **Parkinson**.

### **Estadísticas del Modelo Actual**

| Métrica                   | Valor | ¿Qué significa?                      |
| ------------------------- | ----- | ------------------------------------ |
| **Accuracy**              | 93.8% | Acierta 94 de cada 100 casos         |
| **Precision (Sano)**      | 94.1% | Cuando dice "sano", acierta 94%      |
| **Recall (Sano)**         | 93.5% | Detecta 93.5% de voces sanas         |
| **Precision (Parkinson)** | 93.6% | Cuando dice "Parkinson", acierta 94% |
| **Recall (Parkinson)**    | 94.2% | Detecta 94% de voces con Parkinson   |
| **Tiempo análisis**       | ~1.2s | Muy rápido para uso real             |

### **¿Cuándo entrenar un modelo nuevo?**

**USA EL MODELO ACTUAL si:**

- Solo quieres analizar voces
- Quieres resultados rápidos
- Confías en la precisión del 93.8%

**ENTRENA NUEVO MODELO si:**

- Tienes más datos de entrenamiento
- Quieres personalizar para tu población específica
- Quieres experimentar con la arquitectura

### **¿Dónde se guardan los resultados?**

```
resultados/
└── sesion_20241019_143052/     # Cada análisis en su carpeta
    ├── audio_analizado.wav     # El audio que analizaste
    ├── features_mfcc.npy      # Características extraídas
    ├── features_mel.npy       # Más características
    └── analisis_completo.json # Toda la información técnica
```

### **⚡ RESUMEN RÁPIDO - Solo quiero usarlo**

1. **Instalar**: `pip install -r requirements.txt`
2. **Ejecutar**: `python model/predict_fixed.py`
3. **Elegir opción 1** (grabar) o **2** (archivo)
4. **Ver resultado** y carpeta con detalles

**El modelo ya está entrenado y listo para usar.**

---

## **GUÍA DE INSTALACIÓN PASO A PASO**

### **PASO 1: Requisitos del Sistema**

```bash
# Verificar Python (recomendado 3.10+)
python --version

# Verificar Git
git --version
```

### **PASO 2: Clonar el Proyecto**

```bash
git clone https://github.com/valen725/Neurovoice.git
cd NeuroVoice
```

### **PASO 3: Crear Entorno Virtual**

```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### **PASO 4: Instalar Dependencias**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### **PASO 5: Verificar Instalación**

```bash
python -c "import torch, librosa, sounddevice; print('Instalación exitosa')"
```

---

## **ESCENARIOS DE USO**

### **ESCENARIO A: "Solo quiero probar el sistema"**

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar predictor
python model/predict_fixed.py

# 3. Elegir opción 1 (grabar tu voz) o 2 (archivo)
```

### **ESCENARIO B: "Quiero entrenar mi propio modelo"**

```bash
# 1. Preparar tus datos en data/raw/healthy/ y data/raw/parkinson/

# 2. Ejecutar pipeline completo
python preprocesamiento/01_make_metadata.py --root data/raw --outmeta data/metadata/metadata_master.csv --reportdir data/metadata/reports
python preprocesamiento/02_resample_fix_duration.py --inmeta data/metadata/metadata_master.csv --outmeta data/metadata/metadata_processed.csv --outroot data/processed --sr 16000 --duration 3.0 --vad_db 25
python preprocesamiento/03_extract_features.py --inmeta data/metadata/metadata_processed.csv --outmeta data/metadata/metadata_features.csv --outdir data/features
python preprocesamiento/05_balance_to_target.py --inmeta data/metadata/metadata_processed.csv --outmeta data/metadata/metadata_final_for_training.csv --finalroot data/final_for_training --target_per_class 100 --sr 16000 --duration 3.0 --seed 42

# 3. Entrenar modelo
python model/train.py
```

### **ESCENARIO C: "Quiero evaluar el modelo"**

```bash
python model/evaluate.py
# Genera reportes y gráficos automáticamente
```

---

## **SOLUCIÓN DE PROBLEMAS COMUNES**

### **Error: "No module named 'sounddevice'"**

```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-dev
pip install sounddevice

# macOS
brew install portaudio
pip install sounddevice

# Windows
pip install sounddevice
```

### **Error: "CUDA not available"**

**Solución:** El modelo funciona perfectamente con CPU

```bash
# Verificar que PyTorch funciona
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### **Error: "Permission denied" (micrófono)**

- **Linux**: Verificar permisos de audio
- **macOS**: Configuración > Privacidad > Micrófono
- **Windows**: Configuración > Privacidad > Micrófono

### **Error: "No such file or directory: best_model.pth"**

```bash
# Verificar que el modelo existe
ls model/checkpoints/best_model.pth

# Si no existe, entrenar nuevo modelo
python model/train.py
```

### **❌ Error: "Metadata file not found"**

```bash
# Crear metadata desde archivos existentes
python preprocesamiento/06_fix_metadata.py  # Si tienes el script movido
```

---

## **VERIFICACIÓN DE FUNCIONAMIENTO**

#### **Instalación Básica:**

- [ ] Python 3.8+ instalado
- [ ] Entorno virtual creado y activo
- [ ] Dependencias instaladas sin errores
- [ ] Comando de verificación ejecutado correctamente

#### **Funcionalidad de Predicción:**

- [ ] `python model/predict_fixed.py` se ejecuta
- [ ] Micrófono funciona para grabación
- [ ] Se pueden cargar archivos .wav
- [ ] Resultados se guardan en carpeta `resultados/`

#### **Sistema Completo:**

- [ ] Modelo `best_model.pth` existe
- [ ] Configuración `config.json` es válida
- [ ] Carpeta `data/final_for_training/` tiene archivos
- [ ] Scripts de preprocesamiento ejecutables

---

## **CONSEJOS Y MEJORES PRÁCTICAS**

### **🎤 Para Grabación de Audio:**

- Usar entorno silencioso
- Hablar claramente durante 3 segundos
- Mantener distancia consistente del micrófono
- Probar con diferentes frases (números, palabras largas)

### **Para Archivos de Audio:**

- Formato: `.wav` recomendado
- Duración: 3 segundos (se ajusta automáticamente)
- Calidad: 16kHz mínimo
- Evitar ruido de fondo excesivo

---

## 📈 **INTERPRETACIÓN DE RESULTADOS**

### **Cómo leer las predicciones:**

**Resultado: "Voz SANA (87.3%)"**

- ✅ **Confianza Alta** (>80%): Resultado muy confiable
- **Confianza Media** (60-80%): Resultado moderado, considerar segunda opinión
- ❌ **Confianza Baja** (<60%): Resultado incierto, repetir análisis

**Resultado: "Voz PARKINSON (92.1%)"**

- 🏥 **Recomendación**: Consultar con profesional médico
- **No es diagnóstico**: Solo herramienta de apoyo
- 🔄 **Repetir**: Hacer varias pruebas para consistencia

### **Archivos de Resultados:**

- **`audio_analizado.wav`**: Tu grabación procesada
- **`features_mfcc.npy`**: Características técnicas extraídas
- **`analisis_completo.json`**: Reporte detallado con probabilidades

### **Flujo de Trabajo**

1. **Preprocesamiento**: Scripts 01-05 para preparar datos
2. **Entrenamiento**: `train.py` para entrenar nuevos modelos
3. **Evaluación**: `evaluate.py` para métricas detalladas
4. **Predicción**: `predict_fixed.py` para análisis en tiempo real
5. **Resultados**: Carpeta `resultados/` con análisis organizados

---
