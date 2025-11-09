

import os
import numpy as np
import torch
import librosa
from pathlib import Path
import json
from model.architecture import create_model

def load_default_config():
    """Carga configuración por defecto (idéntico a predict_fixed.py)"""
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
            print(f"Configuración cargada: {path}")
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

def find_best_model():
    """Encuentra el mejor modelo disponible (igual que predict_fixed.py)"""
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


def _load_calibrated_threshold(default: float = 0.3) -> float:
    """Intenta cargar threshold calibrado (best F1) desde JSON.
    Si no existe o falla, retorna default.
    """
    calib_path = Path("model/results/calibration_threshold.json")
    if calib_path.exists():
        try:
            with open(calib_path, 'r') as f:
                data = json.load(f)
            thr = data.get('best_f1_threshold') or data.get('best_youden_threshold')
            if isinstance(thr, (float, int)) and 0.0 < thr < 1.0:
                print(f" Umbral calibrado cargado (F1): {thr:.4f}")
                return float(thr)
        except Exception as e:
            print(f" Error cargando calibración de threshold: {e}")
    return default

def predict_from_file_or_array(model_path, audio_input, config=None, threshold_parkinson=None, robust=False, augmentations=4):
    """Predice desde archivo o array.
    Parámetros:
      threshold_parkinson: umbral base para decidir riesgo.
      robust: si True usa promedio sobre pequeñas variaciones para mayor estabilidad.
      augmentations: número de variantes cuando robust=True.
    """
    predictor = NeuroVoicePredictor(model_path, config)
    if isinstance(audio_input, (str, Path)):
        y, sr = librosa.load(str(audio_input), sr=predictor.sample_rate)
    elif isinstance(audio_input, np.ndarray):
        y = audio_input
    else:
        raise ValueError("audio_input debe ser ruta de archivo o np.ndarray")
    if threshold_parkinson is None:
        threshold_parkinson = _load_calibrated_threshold()
    if robust:
        return predictor.predict_robust(y, threshold_parkinson=threshold_parkinson, n_augment=augmentations)
    return predictor.predict(y, threshold_parkinson=threshold_parkinson)

class NeuroVoicePredictor:
    """Predictor principal de NeuroVoice para la GUI"""
    def __init__(self, model_path, config=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if config is None:
            # Usar la función global load_default_config (idéntica a predict_fixed.py)
            config = load_default_config()
        self.config = config
        self.sample_rate = config.get('sample_rate', 16000)
        self.n_mfcc = config.get('n_mfcc', 20)
        self.n_mels = config.get('n_mels', 128)
        self.label_mapping = {0: 'Healthy', 1: 'Parkinson'}
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = create_model(
            model_type=self.config['architecture'],
            num_classes=2
        )
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()
        return model

    def preprocess_audio(self, audio_data):
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
        window_size = len(audio) // 10
        energies = []
        for i in range(0, len(audio) - window_size, window_size // 2):
            window = audio[i:i + window_size]
            energy = np.sum(window ** 2)
            energies.append(energy)
        if not energies:
            return audio
        max_energy_idx = np.argmax(energies)
        start_sample = max_energy_idx * (window_size // 2)
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
        y = audio_data['audio']
        sr = audio_data['sample_rate']
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc = librosa.util.normalize(mfcc)
        # Mel-spectrograma
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=self.n_mels
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_normalized = librosa.util.normalize(mel_db)
        return {
            'mfcc': mfcc,
            'mel': mel_normalized
        }

    def predict(self, audio_data, threshold_parkinson=None):
        if threshold_parkinson is None:
            threshold_parkinson = _load_calibrated_threshold(0.3)
        processed_audio = self.preprocess_audio(audio_data)
        print(f"[DEBUG] Audio procesado: duración={processed_audio['duration']:.3f}s, max={np.max(np.abs(processed_audio['audio'])):.3f}, min={np.min(processed_audio['audio']):.3f}")
        features = self.extract_features(processed_audio)
        if features is None:
            return None
        mfcc_tensor = torch.FloatTensor(features['mfcc']).unsqueeze(0).unsqueeze(0).to(self.device)
        mel_tensor = torch.FloatTensor(features['mel']).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.config['feature_type'] == 'both':
                outputs = self.model(mfcc_tensor, mel_tensor)
            elif self.config['feature_type'] == 'mfcc':
                outputs = self.model(mfcc_tensor)
            else:
                outputs = self.model(mel_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            probs = probabilities.cpu().numpy()[0]
            parkinson_prob = probs[1]
            healthy_prob = probs[0]
            prob_diff = abs(healthy_prob - parkinson_prob)
        print(f"[DEBUG] Probabilidades: Healthy={healthy_prob:.4f}, Parkinson={parkinson_prob:.4f}, diff={prob_diff:.4f}")
        original_prediction = self.label_mapping[np.argmax(probs)]
        upper = threshold_parkinson + 0.05
        lower = threshold_parkinson - 0.05
        ambiguous_band = (parkinson_prob >= lower) and (parkinson_prob <= upper)
        very_close = prob_diff < 0.15
        if parkinson_prob >= upper:
            final_prediction = "RIESGO ALTO DE PARKINSON"
            confidence_level = "ALTA"
            explanation = (
                "El modelo detecta una probabilidad significativa de Parkinson en la voz. "
                "Se recomienda consultar a un especialista URGENTE."
            )
            final_confidence = parkinson_prob
        elif parkinson_prob >= threshold_parkinson and not very_close:
            final_prediction = "RIESGO MODERADO DE PARKINSON"
            confidence_level = "MEDIA"
            explanation = (
                "El modelo detecta señales compatibles con Parkinson, aunque no son concluyentes. "
                "Se recomienda evaluación médica especializada."
            )
            final_confidence = parkinson_prob
        elif ambiguous_band or very_close:
            final_prediction = "RESULTADO INCIERTO - REPETIR"
            confidence_level = "BAJA"
            explanation = (
                "Zona ambigua: repetir la grabación en ambiente silencioso y mantener vocal constante. "
                "Si persiste, considerar evaluación clínica." 
            )
            final_confidence = max(healthy_prob, parkinson_prob)
        else:
            final_prediction = "VOZ SALUDABLE"
            if prob_diff > 0.6:
                confidence_level = "ALTA"
                explanation = (
                    "El modelo considera que la voz es saludable con alta confianza. "
                    "No se detectan señales relevantes de Parkinson."
                )
            else:
                confidence_level = "MEDIA"
                explanation = (
                    "El modelo considera que la voz es saludable, pero con confianza moderada."
                )
            final_confidence = healthy_prob
        result = {
            'prediction': final_prediction,
            'original_prediction': original_prediction,
            'confidence': float(final_confidence),
            'confidence_level': confidence_level,
            'explanation': explanation,
            'probability_difference': float(prob_diff),
            'parkinson_risk_score': float(parkinson_prob),
            'probabilities': {
                'Healthy': float(probs[0]),
                'Parkinson': float(probs[1])
            }
        }
        return result

    def predict_robust(self, audio_data, threshold_parkinson=None, n_augment=4):
        """Promedia varias predicciones con pequeñas variaciones para mayor estabilidad.
        Mantiene siempre la misma longitud de señal (evita fix_length).
        """
        if threshold_parkinson is None:
            threshold_parkinson = _load_calibrated_threshold(0.3)
        base = self.predict(audio_data, threshold_parkinson=threshold_parkinson)
        if base is None:
            return None
        if n_augment <= 0:
            return base
        probs_h = [base['probabilities']['Healthy']]
        probs_p = [base['probabilities']['Parkinson']]
        N = len(audio_data)
        for _ in range(n_augment):
            aug = audio_data.copy()
            # Ruido gaussiano muy leve
            aug = aug + (np.random.randn(N) * 0.003)
            # Pequeño desplazamiento temporal (roll) que conserva longitud
            shift = int(np.random.uniform(-0.02, 0.02) * N)
            if shift != 0:
                aug = np.roll(aug, shift)
            # Ligera variación de ganancia
            gain = np.random.uniform(0.97, 1.03)
            aug = np.clip(aug * gain, -1.0, 1.0)
            r = self.predict(aug, threshold_parkinson=threshold_parkinson)
            if r is not None:
                probs_h.append(r['probabilities']['Healthy'])
                probs_p.append(r['probabilities']['Parkinson'])
        avg_h = float(np.mean(probs_h))
        avg_p = float(np.mean(probs_p))
        # Re-evaluar decisión con promedios
        combined = self.predict(audio_data, threshold_parkinson=threshold_parkinson)
        if combined is None:
            return None
        combined['probabilities']['Healthy'] = avg_h
        combined['probabilities']['Parkinson'] = avg_p
        combined['parkinson_risk_score'] = avg_p
        combined['probability_difference'] = abs(avg_h - avg_p)
        combined['confidence'] = max(avg_h, avg_p)
        return combined
