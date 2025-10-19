

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuroVoiceCNN(nn.Module):
    """
    Red Neuronal Convolucional para clasificación de Parkinson basada en audio.
    
    Arquitectura híbrida que procesa tanto MFCC como Mel-spectrogramas
    y los combina para la clasificación final.
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(NeuroVoiceCNN, self).__init__()
        
        # === RAMA MFCC (20 x 94) ===
        self.mfcc_branch = nn.Sequential(
            # Bloque Convolucional 1
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (32, 10, 47)
            nn.Dropout2d(dropout_rate * 0.5),
            
            # Bloque Convolucional 2
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (64, 5, 23)
            nn.Dropout2d(dropout_rate * 0.7),
            
            # Bloque Convolucional 3
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 6)),  # (128, 4, 6) - Tamaño fijo
            nn.Dropout2d(dropout_rate),
        )
        
        # === RAMA MEL-SPECTROGRAM (128 x 94) ===
        self.mel_branch = nn.Sequential(
            # Bloque Convolucional 1
            nn.Conv2d(1, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (32, 64, 47)
            nn.Dropout2d(dropout_rate * 0.5),
            
            # Bloque Convolucional 2
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (64, 32, 23)
            nn.Dropout2d(dropout_rate * 0.6),
            
            # Bloque Convolucional 3
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (128, 16, 11)
            nn.Dropout2d(dropout_rate * 0.7),
            
            # Bloque Convolucional 4
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 6)),  # (256, 4, 6) - Tamaño fijo
            nn.Dropout2d(dropout_rate),
        )
        
        # === FUSIÓN Y CLASIFICACIÓN ===
        # MFCC features: 128 * 4 * 6 = 3072
        # Mel features:  256 * 4 * 6 = 6144
        # Total features: 3072 + 6144 = 9216
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(3072 + 6144, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(64, num_classes)
        )
        
        # Inicialización de pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicialización optimizada de pesos."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, mfcc_input, mel_input):
        """
        Forward pass con inputs separados para MFCC y Mel-spectrogram.
        
        Args:
            mfcc_input: Tensor (batch_size, 1, 20, 94)
            mel_input: Tensor (batch_size, 1, 128, 94)
            
        Returns:
            output: Tensor (batch_size, num_classes)
        """
        # Procesar rama MFCC
        mfcc_features = self.mfcc_branch(mfcc_input)
        mfcc_flat = mfcc_features.view(mfcc_features.size(0), -1)
        
        # Procesar rama Mel-spectrogram
        mel_features = self.mel_branch(mel_input)
        mel_flat = mel_features.view(mel_features.size(0), -1)
        
        # Fusionar características
        combined_features = torch.cat([mfcc_flat, mel_flat], dim=1)
        
        # Clasificación final
        output = self.fusion_layers(combined_features)
        
        return output
    
    def get_feature_maps(self, mfcc_input, mel_input):
        """
        Extrae mapas de características para visualización.
        Útil para análisis e interpretabilidad del modelo.
        """
        with torch.no_grad():
            # Features MFCC
            mfcc_conv1 = self.mfcc_branch[0](mfcc_input)
            mfcc_conv2 = self.mfcc_branch[5](F.relu(self.mfcc_branch[4](self.mfcc_branch[3](mfcc_conv1))))
            
            # Features Mel
            mel_conv1 = self.mel_branch[0](mel_input)
            mel_conv2 = self.mel_branch[5](F.relu(self.mel_branch[4](self.mel_branch[3](mel_conv1))))
            
        return {
            'mfcc_conv1': mfcc_conv1,
            'mfcc_conv2': mfcc_conv2,
            'mel_conv1': mel_conv1,
            'mel_conv2': mel_conv2
        }


class NeuroVoiceSimpleCNN(nn.Module):
    """
    Versión simplificada para comparación o cuando hay pocos datos.
    """
    
    def __init__(self, num_classes=2, input_type='mel'):
        super(NeuroVoiceSimpleCNN, self).__init__()
        
        self.input_type = input_type
        
        if input_type == 'mel':
            # Para Mel-spectrogram (128, 94)
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.AdaptiveAvgPool2d((8, 8))
            )
            fc_input_size = 128 * 8 * 8
            
        else:  # MFCC
            # Para MFCC (20, 94)
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 6))
            )
            fc_input_size = 128 * 4 * 6
        
        self.classifier = nn.Sequential(
            nn.Linear(fc_input_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_model(model_type='hybrid', num_classes=2, **kwargs):
    """
    Factory function para crear diferentes variantes del modelo.
    
    Args:
        model_type: 'hybrid', 'mel_only', 'mfcc_only'
        num_classes: Número de clases (2 para sano/enfermo)
        **kwargs: Argumentos adicionales
    
    Returns:
        model: Instancia del modelo
    """
    if model_type == 'hybrid':
        return NeuroVoiceCNN(num_classes=num_classes, **kwargs)
    elif model_type == 'mel_only':
        return NeuroVoiceSimpleCNN(num_classes=num_classes, input_type='mel')
    elif model_type == 'mfcc_only':
        return NeuroVoiceSimpleCNN(num_classes=num_classes, input_type='mfcc')
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")