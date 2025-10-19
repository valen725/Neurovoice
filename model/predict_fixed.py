import torch
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
import json
import war            # Cargar pesos
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # DEBUG: Mostrar información del checkpoint
                if 'epoch' in checkpoint:
                    print(f"🔍 Época del modelo: {checkpoint['epoch']}")
                if 'accuracy' in checkpoint:
                    print(f"🔍 Precisión del modelo: {checkpoint['accuracy']:.4f}")
                if 'loss' in checkpoint:
                    print(f"🔍 Loss del modelo: {checkpoint['loss']:.4f}")
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            print(f"✅ Modelo cargado exitosamente")
            return model time
import os
warnings.filterwarnings('ignore')

# Imports para audio (mismo sistema que datasetGenerator.py)
try:
    import sounddevice as sd
    from scipy.io.wavfile import write
    AUDIO_AVAILABLE = True
    print(" SoundDevice disponible para grabación")
except ImportError:
    AUDIO_AVAILABLE = False
    print(" SoundDevice no disponible. Instala con: pip install sounddevice")

# Imports locales
from architecture import create_model
from utils import load_checkpoint


class SimpleAudioRecorder:
    """Grabador de audio simple usando sounddevice (como datasetGenerator.py)"""
    
    def __init__(self, sample_rate=16000, duration=3):
        self.fs = sample_rate  # Usar mismo nombre que datasetGenerator.py
        self.seconds = duration  # Usar mismo nombre que datasetGenerator.py
        
    def record_audio(self):
        """Graba audio usando el mismo método que datasetGenerator.py"""
        if not AUDIO_AVAILABLE:
            print("❌ SoundDevice no está disponible")
            print("💡 Instala con: pip install sounddevice")
            return None
        
        print(f" Preparando grabación de {self.seconds} segundos...")
        print(f" Frecuencia de muestreo: {self.fs} Hz")
        print("\n Instrucciones:")
        print("   • Ponte cerca del micrófono")
        print("   • Habla con volumen normal")
        print("   • Mantén la vocal 'Ahhhh' durante toda la grabación")
        
        input("\n Presiona ENTER cuando estés listo para grabar...")
        
        try:
            print(" ¡GRABANDO! Habla ahora (mantén la vocal 'a')")
            
            # Usar exactamente el mismo código que datasetGenerator.py
            audio = sd.rec(int(self.seconds * self.fs), samplerate=self.fs, channels=1, dtype='int16')
            sd.wait()
            
            print(" ¡Grabación completada!")
            
            # Convertir a float32 normalizado (como espera el modelo)
            audio_float = audio.flatten().astype(np.float32) / 32768.0
            
            # Verificar calidad
            max_amplitude = np.max(np.abs(audio_float))
            duration_real = len(audio_float) / self.fs
            
            print(f" Duración real: {duration_real:.2f}s")
            print(f" Nivel máximo: {max_amplitude:.3f}")
            
            if max_amplitude < 0.01:
                print(" Audio muy silencioso, habla más fuerte")
            elif max_amplitude > 0.95:
                print(" Audio muy fuerte, puede haber distorsión")
            else:
                print(" Nivel de audio correcto")
            
            # Guardar copia del audio grabado
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            temp_file = f"model/recorded_audio_{timestamp}.wav"
            
            # Crear directorio si no existe
            os.makedirs("model", exist_ok=True)
            
            # Guardar usando scipy (mismo método que datasetGenerator.py)
            write(temp_file, self.fs, audio)
            print(f" Audio guardado: {temp_file}")
            
            return audio_float
            
        except Exception as e:
            print(f" Error al grabar: {e}")
            return None


class NeuroVoicePredictor:
    """Predictor principal de NeuroVoice"""
    
    def __init__(self, model_path, config=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f" Using device: {self.device}")
        
        # Cargar configuración
        if config is None:
            config = self.load_default_config()
        
        self.config = config
        self.sample_rate = config.get('sample_rate', 16000)
        self.n_mfcc = config.get('n_mfcc', 20)
        self.n_mels = config.get('n_mels', 128)
        
        # Mapeo de etiquetas
        self.label_mapping = {0: 'Healthy', 1: 'Parkinson'}
        
        # Cargar modelo
        print(f" Loading model from {model_path}")
        self.model = self.load_model(model_path)
        
    def load_default_config(self):
        """Carga configuración por defecto"""
        return {
            'model_type': 'hybrid',
            'architecture': 'hybrid',
            'sample_rate': 16000,
            'n_mfcc': 20,
            'n_mels': 128,
            'feature_type': 'both'
        }
    
    def load_model(self, model_path):
        """Carga el modelo entrenado"""
        try:
            # Crear modelo (sin pasar parámetros no soportados)
            model = create_model(
                model_type=self.config['architecture'],
                num_classes=2
            )
            
            # Cargar pesos
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            print(f"Modelo cargado exitosamente")
            return model
            
        except Exception as e:
            raise Exception(f"Error cargando modelo: {e}")
    
    def preprocess_audio(self, audio_data):
        """Preprocesa el audio para el modelo"""
        y = audio_data
        sr = self.sample_rate
        
        # Aplicar VAD básico (remover silencios)
        y = self.apply_simple_vad(y)
        
        # Normalizar
        y = self.normalize_audio(y)
        
        # Ajustar a 3 segundos exactos
        target_length = sr * 3
        if len(y) > target_length:
            # Recortar desde el centro
            start = (len(y) - target_length) // 2
            y = y[start:start + target_length]
        elif len(y) < target_length:
            # Pad con zeros
            pad_needed = target_length - len(y)
            pad_left = pad_needed // 2
            pad_right = pad_needed - pad_left
            y = np.pad(y, (pad_left, pad_right), mode='constant')
        
        return {
            'audio': y,
            'sample_rate': sr,
            'duration': len(y) / sr
        }
    
    def apply_simple_vad(self, audio, threshold=0.01):
        """VAD simple: encuentra la parte con más actividad"""
        # Calcular energía en ventanas
        window_size = len(audio) // 10
        energies = []
        
        for i in range(0, len(audio) - window_size, window_size // 2):
            window = audio[i:i + window_size]
            energy = np.sum(window ** 2)
            energies.append(energy)
        
        if not energies:
            return audio
        
        # Encontrar ventana con más energía
        max_energy_idx = np.argmax(energies)
        start_sample = max_energy_idx * (window_size // 2)
        
        # Expandir alrededor del punto de máxima energía
        expand_samples = self.sample_rate * 2  # 2 segundos alrededor
        start = max(0, start_sample - expand_samples // 2)
        end = min(len(audio), start_sample + expand_samples // 2)
        
        return audio[start:end] if end > start else audio
    
    def normalize_audio(self, audio):
        """Normaliza el audio"""
        if np.max(np.abs(audio)) > 0:
            return audio / np.max(np.abs(audio)) * 0.95
        return audio
    
    def extract_features(self, audio_data):
        """Extrae características MFCC y Mel-spectrograma"""
        try:
            y = audio_data['audio']
            sr = audio_data['sample_rate']
            
            print("🧮 Extrayendo características...")
            
            # MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfcc = librosa.util.normalize(mfcc)
            
            # Mel-spectrograma
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=self.n_mels
            )
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_normalized = librosa.util.normalize(mel_db)
            
            print(f"   MFCC shape: {mfcc.shape}")
            print(f"   Mel shape: {mel_normalized.shape}")
            
            # DEBUG: Estadísticas de las características
            print(f"   MFCC stats - min: {mfcc.min():.3f}, max: {mfcc.max():.3f}, mean: {mfcc.mean():.3f}")
            print(f"   Mel stats - min: {mel_normalized.min():.3f}, max: {mel_normalized.max():.3f}, mean: {mel_normalized.mean():.3f}")
            
            return {
                'mfcc': mfcc,
                'mel': mel_normalized
            }
            
        except Exception as e:
            print(f"❌ Error extrayendo características: {e}")
            return None
    
    def predict_live_audio(self, audio_data):
        """Predice para audio grabado en vivo"""
        print("\n Analizando audio...")
        
        # Preprocesar
        processed_audio = self.preprocess_audio(audio_data)
        
        # Extraer características
        features = self.extract_features(processed_audio)
        if features is None:
            return None
        
        # Preparar tensores
        mfcc_tensor = torch.FloatTensor(features['mfcc']).unsqueeze(0).unsqueeze(0)
        mel_tensor = torch.FloatTensor(features['mel']).unsqueeze(0).unsqueeze(0)
        
        mfcc_tensor = mfcc_tensor.to(self.device)
        mel_tensor = mel_tensor.to(self.device)
        
        # Predicción
        with torch.no_grad():
            if self.config['feature_type'] == 'both':
                outputs = self.model(mfcc_tensor, mel_tensor)
            elif self.config['feature_type'] == 'mfcc':
                outputs = self.model(mfcc_tensor)
            else:  # mel
                outputs = self.model(mel_tensor)
            
            # DEBUG: Mostrar outputs crudos
            print(f"🔍 Raw outputs: {outputs}")
            print(f"🔍 Output shape: {outputs.shape}")
            
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            prediction = predicted.cpu().numpy()[0]
            probs = probabilities.cpu().numpy()[0]
            
            # DEBUG: Mostrar más detalles
            print(f"🔍 Predicted class index: {prediction}")
            print(f"🔍 Raw probabilities: {probs}")
            print(f"🔍 Softmax sum: {np.sum(probs):.6f}")
        
        # Interpretar resultados
        predicted_class = self.label_mapping[prediction]
        confidence = probs[prediction]
        
        result = {
            'prediction': predicted_class,
            'confidence': float(confidence),
            'probabilities': {
                'Healthy': float(probs[0]),
                'Parkinson': float(probs[1])
            }
        }
        
        return result
    
    def predict_audio_file(self, audio_path):
        """Predice para archivo de audio"""
        try:
            print(f" Analizando archivo: {Path(audio_path).name}")
            
            # Cargar audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Predecir
            return self.predict_live_audio(audio)
            
        except Exception as e:
            print(f" Error procesando archivo: {e}")
            return None


def find_best_model():
    """Encuentra el mejor modelo disponible"""
    model_paths = [
        "model/checkpoints/best_model.pth",
        "model/best_model.pth",
        "checkpoints/best_model.pth",
        "best_model.pth"
    ]
    
    for path in model_paths:
        if Path(path).exists():
            print(f" Modelo encontrado: {path}")
            return str(Path(path).absolute())
    
    print(" No se encontró modelo entrenado")
    return None


def load_default_config():
    """Carga configuración por defecto"""
    config_paths = [
        "model/config.json",
        "config.json"
    ]
    
    for path in config_paths:
        if Path(path).exists():
            with open(path, 'r') as f:
                nested_config = json.load(f)
            
            # Convertir configuración anidada a plana para compatibilidad
            flat_config = {
                'model_type': 'hybrid',
                'architecture': 'hybrid',
                'sample_rate': nested_config.get('data', {}).get('target_sample_rate', 16000),
                'n_mfcc': nested_config.get('features', {}).get('mfcc', {}).get('n_mfcc', 20),
                'n_mels': nested_config.get('features', {}).get('mel', {}).get('n_mels', 128),
                'feature_type': 'both'
            }
            
            print(f"✅ Configuración cargada: {path}")
            return flat_config
    
    print(" Usando configuración por defecto")
    return {
        'model_type': 'hybrid',
        'architecture': 'hybrid', 
        'sample_rate': 16000,
        'n_mfcc': 20,
        'n_mels': 128,
        'feature_type': 'both'
    }


def main():
    """Función principal"""
    print(" NeuroVoice - Detector de Parkinson por Voz")
    print("=" * 50)
    
    # Buscar modelo automáticamente
    print(" Buscando modelo entrenado...")
    model_path = find_best_model()
    if not model_path:
        print(" Entrena un modelo primero con: python model/train.py")
        return
    
    # Cargar configuración
    config = load_default_config()
    
    # Crear predictor
    try:
        predictor = NeuroVoicePredictor(model_path, config)
        print(" Sistema listo para análisis")
    except Exception as e:
        print(f"Error inicializando sistema: {e}")
        return
    
    # Crear grabador (usando mismo sistema que datasetGenerator.py)
    recorder = SimpleAudioRecorder(sample_rate=16000, duration=3)
    
    print("=" * 50)
    
    # Loop principal
    while True:
        print("\n🎵 Opciones:")
        if AUDIO_AVAILABLE:
            print("1.  Grabar y analizar mi voz")
            print("2.  Analizar archivo de audio")
            print("3.  Salir")
        else:
            print("1.  Analizar archivo de audio")
            print("2.  Salir")
            print(" (Grabación no disponible - instala sounddevice)")
    
        choice = input("\nSelecciona una opción: ").strip()
        
        if AUDIO_AVAILABLE and choice == "1":
            # Grabación en vivo
            print("\nGRABACIÓN EN DIRECTO")
            print("=" * 30)
            
            audio_data = recorder.record_audio()
            
            if audio_data is not None:
                result = predictor.predict_live_audio(audio_data)
                
                if result:
                    print("\n📊 RESULTADO DEL ANÁLISIS")
                    print("=" * 30)
                    
                    if result['prediction'] == 'Healthy':
                        print("El análisis sugiere voz saludable")
                    else:
                        print(" El análisis detecta posibles indicios de Parkinson")
                    
                    print(f"\n Confianza: {result['confidence']:.1%}")
                    print(f"Probabilidades:")
                    print(f"   Healthy: {result['probabilities']['Healthy']:.1%}")
                    print(f"   Parkinson: {result['probabilities']['Parkinson']:.1%}")
                    
                    # Guardar resultado
                    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                    result_file = f"model/analysis_{timestamp}.json"
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=4)
                    print(f"\n Resultado guardado: {result_file}")
                    
                    print("\nIMPORTANTE: Consulta con un médico profesional")
        
        elif (AUDIO_AVAILABLE and choice == "2") or (not AUDIO_AVAILABLE and choice == "1"):
            # Análisis de archivo
            audio_path = input(" Ruta del archivo de audio: ").strip()
            if audio_path and Path(audio_path).exists():
                result = predictor.predict_audio_file(audio_path)
                
                if result:
                    print("\n RESULTADO DEL ANÁLISIS")
                    print("=" * 30)
                    
                    if result['prediction'] == 'Healthy':
                        print("El análisis sugiere voz saludable")
                    else:
                        print(" El análisis detecta posibles indicios de Parkinson")
                    
                    print(f"\nConfianza: {result['confidence']:.1%}")
                    print(f"Probabilidades:")
                    print(f"Healthy: {result['probabilities']['Healthy']:.1%}")
                    print(f"Parkinson: {result['probabilities']['Parkinson']:.1%}")
                    
                    print("\n IMPORTANTE: Consulta con un médico profesional")
            else:
                print(" Archivo no encontrado")
        
        elif (AUDIO_AVAILABLE and choice == "3") or (not AUDIO_AVAILABLE and choice == "2"):
            print("\n ¡Gracias por usar NeuroVoice!")
            break
        else:
            print(" Opción inválida")
        
        input("\nPresiona ENTER para continuar...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n ¡Hasta luego!")
    except Exception as e:
        print(f"\nError: {e}")
        print("Verifica que sounddevice esté instalado: pip install sounddevice")
