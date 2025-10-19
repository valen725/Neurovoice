#!/bin/bash

# Script de instalación para NeuroVoice Live Recording
echo "🔧 Instalando dependencias para grabación en directo..."

# Instalar pyaudio (requerido para grabación de audio)
echo "📦 Instalando PyAudio..."

# Para Ubuntu/Debian
if command -v apt-get &> /dev/null; then
    echo "🐧 Detectado sistema Debian/Ubuntu"
    sudo apt-get update
    sudo apt-get install -y portaudio19-dev python3-pyaudio
    pip install pyaudio
elif command -v brew &> /dev/null; then
    echo "🍎 Detectado macOS"
    brew install portaudio
    pip install pyaudio
else
    echo "🐧 Instalando PyAudio con pip..."
    pip install pyaudio
fi

echo "✅ Instalación completada"
echo "🎤 Ya puedes usar la grabación en directo con:"
echo "   python model/predict_fixed.py"