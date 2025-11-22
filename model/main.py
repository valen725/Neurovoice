from predict_fixed import NeuroVoicePredictor, load_default_config, find_best_model, SimpleAudioRecorder

import pandas as pd
import json
from pathlib import Path
AUDIO_AVAILABLE = True


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
        print("\n Opciones:")
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
                    print(f"\n RESULTADO DEL ANÁLISIS AVANZADO")
                    print("=" * 40)
                    
                    # Mostrar resultado principal con colores/símbolos
                    if "Healthy" in result['prediction']:
                        print(f" {result['prediction']}")
                    elif "RIESGO ALTO" in result['prediction']:
                        print(f" {result['prediction']}")
                    elif "RIESGO MODERADO" in result['prediction']:
                        print(f" {result['prediction']}")
                    elif "INCIERTO" in result['prediction']:
                        print(f" {result['prediction']}")
                    else:
                        print(f" {result['prediction']}")
                    
                    print(f"\n Confianza: {result['confidence']:.1%}")
                    print(f" Nivel de confianza: {result['confidence_level']}")
                    print(f" Score de riesgo Parkinson: {result['parkinson_risk_score']:.1%}")
                    print(f" Explicación: {result['explanation']}")
                    print(f" Diferencia entre probabilidades: {result['probability_difference']:.1%}")
                    
                    print(f"\nProbabilidades detalladas:")
                    print(f"   Healthy: {result['probabilities']['Healthy']:.1%}")
                    print(f"   Parkinson: {result['probabilities']['Parkinson']:.1%}")
                
                    
                    # Guardar resultado
                    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                    result_file = f"model/analysis_{timestamp}.json"
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=4)
                    print(f"\n Resultado guardado: {result_file}")
                    
                    print("\nIMPORTANTE: Este análisis es solo una herramienta de apoyo")
        
        elif (AUDIO_AVAILABLE and choice == "2") or (not AUDIO_AVAILABLE and choice == "1"):
            # Análisis de archivo
            audio_path = input(" Ruta del archivo de audio: ").strip()
            if audio_path and Path(audio_path).exists():
                result = predictor.predict_audio_file(audio_path)
                
                if result:
                    print(f"\n RESULTADO DEL ANÁLISIS AVANZADO")
                    print("=" * 40)
                    
                    # Mostrar resultado principal con colores/símbolos
                    if "Healthy" in result['prediction']:
                        print(f" {result['prediction']}")
                    elif "RIESGO ALTO" in result['prediction']:
                        print(f" {result['prediction']}")
                    elif "RIESGO MODERADO" in result['prediction']:
                        print(f"{result['prediction']}")
                    elif "INCIERTO" in result['prediction']:
                        print(f" {result['prediction']}")
                    else:
                        print(f" {result['prediction']}")
                    
                    print(f"\n Confianza: {result['confidence']:.1%}")
                    print(f" Nivel de confianza: {result['confidence_level']}")
                    print(f" Score de riesgo Parkinson: {result['parkinson_risk_score']:.1%}")
                    print(f" Explicación: {result['explanation']}")
                    print(f" Diferencia entre probabilidades: {result['probability_difference']:.1%}")
                    
                    print(f"\n Probabilidades detalladas:")
                    print(f"   Healthy: {result['probabilities']['Healthy']:.1%}")
                    print(f"   Parkinson: {result['probabilities']['Parkinson']:.1%}")
                    
                    if 'original_prediction' in result:
                        print(f"\n Predicción original del modelo: {result['original_prediction']}")
                    
                    print("\n IMPORTANTE: Este análisis es solo una herramienta de apoyo")
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
        print("Usted es una mierda")
