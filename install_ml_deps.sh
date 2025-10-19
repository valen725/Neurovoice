#!/bin/bash

# 🧠 NeuroVoice - Instalador de Dependencias de Machine Learning
# Script para instalar PyTorch y dependencias adicionales

echo "🚀 Instalando dependencias de Machine Learning para NeuroVoice..."

# Verificar que estamos en el entorno virtual
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  No se detectó entorno virtual activo"
    echo "   Asegúrate de activar tu entorno virtual primero:"
    echo "   source .venv/bin/activate"
    exit 1
fi

echo "✅ Entorno virtual detectado: $VIRTUAL_ENV"

# Instalar PyTorch (CPU version para compatibilidad)
echo "📦 Instalando PyTorch..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Instalar scikit-learn y seaborn
echo "📦 Instalando scikit-learn y seaborn..."
pip install scikit-learn==1.3.2 seaborn==0.12.2

# Verificar instalación
echo "🔍 Verificando instalación..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__} instalado correctamente')"
python -c "import sklearn; print(f'✅ Scikit-learn {sklearn.__version__} instalado correctamente')"
python -c "import seaborn; print(f'✅ Seaborn instalado correctamente')"

echo ""
echo "🎉 ¡Instalación completa!"
echo "📋 Dependencias instaladas:"
echo "   - PyTorch 2.1.0 (CPU)"
echo "   - TorchVision 0.16.0"
echo "   - Scikit-learn 1.3.2"
echo "   - Seaborn 0.12.2"
echo ""
echo "🧠 NeuroVoice está listo para entrenar modelos!"
echo "💡 Siguiente paso: python model/train.py"