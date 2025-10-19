#!/usr/bin/env python3
"""
Script para crear metadata completo con todos los archivos de audio
"""

import pandas as pd
from pathlib import Path
import os

def create_complete_metadata():
    """Crea metadata completo con todos los archivos de entrenamiento"""
    
    print(" Creando metadata completo...")
    
    # Lista para almacenar los datos
    data = []
    
    # Directorio base
    base_dir = Path("data/final_for_training")
    
    # Procesar archivos healthy
    healthy_dir = base_dir / "healthy"
    if healthy_dir.exists():
        print(f"Procesando directorio: {healthy_dir}")
        wav_files = list(healthy_dir.glob("*.wav"))
        print(f"   Encontrados {len(wav_files)} archivos healthy")
        
        for wav_file in wav_files:
            data.append({
                'filepath': f'data/final_for_training/healthy/{wav_file.name}',
                'label': 'healthy',
                'filename': wav_file.name,
                'source': 'balanced_dataset',
                'valid': True,
                'sample_rate': 16000,
                'duration_sec': 3.0
            })
    else:
        print(f" No se encuentra directorio: {healthy_dir}")
    
    # Procesar archivos parkinson
    parkinson_dir = base_dir / "parkinson"
    if parkinson_dir.exists():
        print(f"📁 Procesando directorio: {parkinson_dir}")
        wav_files = list(parkinson_dir.glob("*.wav"))
        print(f"   Encontrados {len(wav_files)} archivos parkinson")
        
        for wav_file in wav_files:
            data.append({
                'filepath': f'data/final_for_training/parkinson/{wav_file.name}',
                'label': 'parkinson',
                'filename': wav_file.name,
                'source': 'balanced_dataset',
                'valid': True,
                'sample_rate': 16000,
                'duration_sec': 3.0
            })
    else:
        print(f"❌ No se encuentra directorio: {parkinson_dir}")
    
    # Crear DataFrame
    df = pd.DataFrame(data)
    
    # Mostrar estadísticas
    print(f"\nEstadísticas del dataset:")
    print(f"   Total de muestras: {len(df)}")
    print(f"   Healthy: {len(df[df['label'] == 'healthy'])}")
    print(f"   Parkinson: {len(df[df['label'] == 'parkinson'])}")
    
    # Guardar el archivo
    output_file = "data/metadata/metadata_final_for_training_complete.csv"
    df.to_csv(output_file, index=False)
    print(f"\nMetadata guardado en: {output_file}")
    
    # Reemplazar el archivo original
    original_file = "data/metadata/metadata_final_for_training.csv"
    backup_file = "data/metadata/metadata_final_for_training_backup.csv"
    
    # Hacer backup del original
    if os.path.exists(original_file):
        print(f" Creando backup: {backup_file}")
        os.rename(original_file, backup_file)
    
    # Copiar el nuevo archivo como el oficial
    print(f" Actualizando archivo oficial: {original_file}")
    df.to_csv(original_file, index=False)
    
    print("\n ¡Metadata completo creado exitosamente!")
    print(f"   Archivo principal: {original_file}")
    print(f"   Backup anterior: {backup_file}")
    print(f"   Archivo completo: {output_file}")
    
    return df

if __name__ == "__main__":
    create_complete_metadata()