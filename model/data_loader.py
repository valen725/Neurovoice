
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
from typing import Tuple, Dict, Optional, List


class NeuroVoiceDataset(Dataset):
    """
    Dataset personalizado para NeuroVoice.
    Carga y procesa features MFCC y Mel-spectrogramas.
    """
    
    def __init__(self, 
                 metadata_file: str,
                 transform=None,
                 augment=False,
                 feature_type='both',
                 verbose=False):
        """
        Args:
            metadata_file: Ruta al archivo CSV con metadata
            transform: Transformaciones a aplicar
            augment: Si aplicar augmentación de datos
            feature_type: 'both', 'mfcc', 'mel'
            verbose: Si mostrar información detallada
        """
        self.df = pd.read_csv(metadata_file)
        self.transform = transform
        self.augment = augment
        self.feature_type = feature_type
        
        # Encoder para labels
        self.label_encoder = LabelEncoder()
        self.df['label_encoded'] = self.label_encoder.fit_transform(self.df['label'])
        
        # Mapping de labels
        self.label_mapping = {
            'healthy': 0,
            'parkinson': 1
        }
        
        # Información del dataset
        self.num_classes = len(self.label_encoder.classes_)
        self.class_names = self.label_encoder.classes_
        
        if verbose:
            print(f"Dataset initialized: {len(self.df)} samples, {len(self.class_names)} classes")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Retorna una muestra del dataset.
        
        Returns:
            Si feature_type='both': (mfcc_tensor, mel_tensor, label)
            Si feature_type='mfcc': (mfcc_tensor, label)
            Si feature_type='mel': (mel_tensor, label)
        """
        row = self.df.iloc[idx]
        label = row['label_encoded']
        
        try:
            if self.feature_type in ['both', 'mfcc']:
                mfcc = np.load(row['mfcc_path'])
                mfcc = self._process_mfcc(mfcc)
                
            if self.feature_type in ['both', 'mel']:
                mel = np.load(row['mel_path'])
                mel = self._process_mel(mel)
            
            # Aplicar augmentación si está habilitada
            if self.augment:
                if self.feature_type in ['both', 'mfcc']:
                    mfcc = self._augment_features(mfcc)
                if self.feature_type in ['both', 'mel']:
                    mel = self._augment_features(mel)
            
            # Aplicar transformaciones adicionales
            if self.transform:
                if self.feature_type in ['both', 'mfcc']:
                    mfcc = self.transform(mfcc)
                if self.feature_type in ['both', 'mel']:
                    mel = self.transform(mel)
            
            # Retornar según el tipo de feature
            if self.feature_type == 'both':
                return torch.FloatTensor(mfcc), torch.FloatTensor(mel), torch.LongTensor([label])
            elif self.feature_type == 'mfcc':
                return torch.FloatTensor(mfcc), torch.LongTensor([label])
            elif self.feature_type == 'mel':
                return torch.FloatTensor(mel), torch.LongTensor([label])
                
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Retornar sample por defecto en caso de error
            if self.feature_type == 'both':
                return (torch.zeros(1, 20, 94), torch.zeros(1, 128, 94), torch.LongTensor([0]))
            elif self.feature_type == 'mfcc':
                return (torch.zeros(1, 20, 94), torch.LongTensor([0]))
            elif self.feature_type == 'mel':
                return (torch.zeros(1, 128, 94), torch.LongTensor([0]))
    
    def _process_mfcc(self, mfcc):
        """Procesa features MFCC."""
        # Normalizar y agregar dimensión de canal
        mfcc = np.expand_dims(mfcc, axis=0)  # (1, 20, 94)
        return mfcc
    
    def _process_mel(self, mel):
        """Procesa Mel-spectrogramas."""
        # Normalizar y agregar dimensión de canal
        mel = np.expand_dims(mel, axis=0)  # (1, 128, 94)
        return mel
    
    def _augment_features(self, features):
        """
        Aplica augmentación de datos a los features.
        """
        augmented = features.copy()
        
        # Time masking (enmascarar columnas de tiempo)
        if random.random() < 0.3:
            mask_size = random.randint(1, 5)
            start_col = random.randint(0, features.shape[2] - mask_size)
            augmented[:, :, start_col:start_col + mask_size] = 0
        
        # Frequency masking (enmascarar filas de frecuencia)
        if random.random() < 0.3:
            mask_size = random.randint(1, min(8, features.shape[1]))
            start_row = random.randint(0, features.shape[1] - mask_size)
            augmented[:, start_row:start_row + mask_size, :] = 0
        
        # Gaussian noise
        if random.random() < 0.2:
            noise = np.random.normal(0, 0.01, features.shape)
            augmented = augmented + noise
        
        # Volume scaling
        if random.random() < 0.3:
            scale = random.uniform(0.8, 1.2)
            augmented = augmented * scale
        
        return augmented
    
    def get_class_weights(self):
        """Calcula pesos de clase para balancear el entrenamiento."""
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.df['label_encoded']),
            y=self.df['label_encoded']
        )
        return torch.FloatTensor(class_weights)
    
    def get_sample_info(self, idx):
        """Retorna información detallada de una muestra."""
        row = self.df.iloc[idx]
        return {
            'index': idx,
            'filename': row['filename'],
            'label': row['label'],
            'label_encoded': row['label_encoded'],
            'mfcc_path': row.get('mfcc_path', 'N/A'),
            'mel_path': row.get('mel_path', 'N/A')
        }


def create_data_loaders(metadata_file: str,
                       batch_size: int = 32,
                       test_size: float = 0.2,
                       validation_size: float = 0.2,
                       feature_type: str = 'both',
                       augment_train: bool = True,
                       random_state: int = 42,
                       num_workers: int = 4,
                       verbose: bool = True) -> Dict[str, DataLoader]:
    """
    Crea data loaders para entrenamiento, validación y test.
    
    Args:
        metadata_file: Ruta al archivo CSV con metadata
        batch_size: Tamaño del batch
        test_size: Proporción para test set
        validation_size: Proporción para validation set
        feature_type: 'both', 'mfcc', 'mel'
        augment_train: Si aplicar augmentación solo en train
        random_state: Seed para reproducibilidad
        num_workers: Número de workers para carga de datos
        verbose: Si mostrar información de división de datos
        
    Returns:
        Dict con data loaders: {'train', 'val', 'test'}
    """
    
    # Cargar metadata
    df = pd.read_csv(metadata_file)
    
    # Separar train+val de test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['label']
    )
    
    # Separar train de validation
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=validation_size,
        random_state=random_state,
        stratify=train_val_df['label']
    )
    
    if verbose:
        print(f"Data Split: Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    # Guardar splits temporales
    train_df.to_csv('temp_train.csv', index=False)
    val_df.to_csv('temp_val.csv', index=False)
    test_df.to_csv('temp_test.csv', index=False)
    
    # Crear datasets
    train_dataset = NeuroVoiceDataset(
        'temp_train.csv',
        feature_type=feature_type,
        augment=augment_train,
        verbose=False
    )
    
    val_dataset = NeuroVoiceDataset(
        'temp_val.csv',
        feature_type=feature_type,
        augment=False,
        verbose=False
    )
    
    test_dataset = NeuroVoiceDataset(
        'temp_test.csv',
        feature_type=feature_type,
        augment=False,
        verbose=False
    )
    
    # Crear data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Limpiar archivos temporales
    import os
    os.remove('temp_train.csv')
    os.remove('temp_val.csv')
    os.remove('temp_test.csv')
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'datasets': {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
    }


def analyze_dataset(metadata_file: str):
    """
    Analiza el dataset y muestra estadísticas útiles.
    """
    print("Analyzing dataset...")
    
    df = pd.read_csv(metadata_file)
    
    print(f"\nGeneral information:")
    print(f"   Total samples: {len(df)}")
    print(f"   Classes: {df['label'].unique()}")
    print(f"   Distribution by class:")
    for label, count in df['label'].value_counts().items():
        print(f"     {label}: {count} ({count/len(df)*100:.1f}%)")
    
    # Verificar que los archivos existen
    missing_mfcc = 0
    missing_mel = 0
    
    for _, row in df.iterrows():
        if not Path(row['mfcc_path']).exists():
            missing_mfcc += 1
        if not Path(row['mel_path']).exists():
            missing_mel += 1
    
    print(f"\nFile verification:")
    print(f"   MFCC files missing: {missing_mfcc}")
    print(f"   Mel files missing: {missing_mel}")
    
    if missing_mfcc == 0 and missing_mel == 0:
        print("   All files are available")
    
    # Cargar sample para verificar dimensiones
    try:
        sample_mfcc = np.load(df.iloc[0]['mfcc_path'])
        sample_mel = np.load(df.iloc[0]['mel_path'])
        
        print(f"\nFeature dimensions:")
        print(f"   MFCC: {sample_mfcc.shape}")
        print(f"   Mel-spectrogram: {sample_mel.shape}")
        
    except Exception as e:
        print(f"Error loading sample: {e}")


if __name__ == "__main__":
    # Test del data loader
    print("Testing NeuroVoice Data Loader...")
    
    metadata_file = "data/metadata/metadata_features_final.csv"
    
    if Path(metadata_file).exists():
        # Analizar dataset
        analyze_dataset(metadata_file)
        
        # Crear data loaders
        data_loaders = create_data_loaders(
            metadata_file=metadata_file,
            batch_size=8,
            feature_type='both'
        )
        
        # Test de un batch
        train_loader = data_loaders['train']
        for batch_idx, (mfcc, mel, labels) in enumerate(train_loader):
            print(f"\nBatch {batch_idx}:")
            print(f"   MFCC shape: {mfcc.shape}")
            print(f"   Mel shape: {mel.shape}")
            print(f"   Labels shape: {labels.shape}")
            print(f"   Labels: {labels.flatten().tolist()}")
            break
            
    else:
        print(f"Metadata file not found: {metadata_file}")
        print("   Run the feature extraction pipeline first.")