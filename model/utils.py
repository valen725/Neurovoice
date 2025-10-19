

import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from pathlib import Path
from typing import Dict, List, Any
import time


class EarlyStopping:
    """
    Early stopping para evitar overfitting.
    """
    
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        """
        Args:
            patience: Número de epochs sin mejora antes de parar
            min_delta: Cambio mínimo para considerar como mejora
            mode: 'max' para métricas que queremos maximizar, 'min' para minimizar
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.wait = 0
        self.best_score = None
        self.stopped = False
    
    def __call__(self, score):
        """
        Returns:
            True si debemos parar el entrenamiento
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped = True
            return True
        
        return False


class MetricsTracker:
    """
    Tracker para métricas de entrenamiento y validación.
    """
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_auc': [],
            'epochs': [],
            'learning_rates': []
        }
    
    def update(self, epoch, train_metrics, val_metrics):
        """Actualiza las métricas del epoch actual."""
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_accuracy'].append(train_metrics['accuracy'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_accuracy'].append(val_metrics['accuracy'])
        self.history['val_auc'].append(val_metrics['auc'])
    
    def get_history(self):
        """Retorna el historial de métricas."""
        return self.history
    
    def save_history(self, filepath):
        """Guarda el historial en archivo JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def load_history(self, filepath):
        """Carga historial desde archivo JSON."""
        with open(filepath, 'r') as f:
            self.history = json.load(f)


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    """
    Guarda checkpoint del modelo.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'timestamp': time.time()
    }
    
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model=None, optimizer=None, scheduler=None):
    """
    Carga checkpoint del modelo.
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


class ModelEvaluator:
    """
    Evaluador completo para el modelo entrenado.
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def evaluate_dataset(self, data_loader, class_names=None):
        """
        Evalúa el modelo en un dataset completo.
        
        Returns:
            Dict con métricas y predicciones
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_data in data_loader:
                # Procesar batch según el tipo de features
                if len(batch_data) == 3:  # both features
                    mfcc, mel, labels = batch_data
                    mfcc, mel = mfcc.to(self.device), mel.to(self.device)
                    labels = labels.to(self.device).squeeze()
                    outputs = self.model(mfcc, mel)
                else:  # single feature type
                    features, labels = batch_data
                    features = features.to(self.device)
                    labels = labels.to(self.device).squeeze()
                    outputs = self.model(features)
                
                # Calcular loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                # Predicciones
                _, predicted = torch.max(outputs.data, 1)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Acumular resultados
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calcular métricas
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(data_loader)
        
        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc_score = roc_auc_score(all_labels, np.array(all_probabilities)[:, 1])
        except:
            auc_score = 0.0
        
        # Classification report
        if class_names is None:
            class_names = ['Healthy', 'Parkinson']
        
        report = classification_report(
            all_labels, 
            all_predictions, 
            target_names=class_names,
            output_dict=True
        )
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'auc': auc_score,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'classification_report': report
        }
    
    def generate_confusion_matrix(self, labels, predictions, class_names=None, save_path=None):
        """Genera y guarda matriz de confusión."""
        if class_names is None:
            class_names = ['Healthy', 'Parkinson']
        
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return cm
    
    def generate_roc_curve(self, labels, probabilities, save_path=None):
        """Genera curva ROC."""
        fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fpr, tpr, roc_auc


def plot_training_history(history_file, save_dir=None):
    """
    Plotea el historial de entrenamiento desde archivo JSON.
    """
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy')
    axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # AUC
    axes[1, 0].plot(epochs, history['val_auc'], 'g-', label='Validation AUC')
    axes[1, 0].set_title('Validation AUC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate (si está disponible)
    if 'learning_rates' in history and history['learning_rates']:
        axes[1, 1].plot(epochs, history['learning_rates'], 'purple', label='Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Training history plot saved to {save_path}")
        plt.close()
    else:
        plt.show()


def analyze_model_predictions(labels, predictions, probabilities, class_names=None, save_dir=None):
    """
    Análisis completo de las predicciones del modelo.
    """
    if class_names is None:
        class_names = ['Healthy', 'Parkinson']
    
    results = {}
    
    # Métricas básicas
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    results['accuracy'] = accuracy_score(labels, predictions)
    results['precision'] = precision_score(labels, predictions, average='weighted')
    results['recall'] = recall_score(labels, predictions, average='weighted')
    results['f1_score'] = f1_score(labels, predictions, average='weighted')
    
    # Por clase
    results['per_class'] = classification_report(labels, predictions, 
                                               target_names=class_names, output_dict=True)
    
    # Matriz de confusión
    cm = confusion_matrix(labels, predictions)
    results['confusion_matrix'] = cm.tolist()
    
    # Visualizaciones
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(save_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Distribución de probabilidades
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        for i, class_name in enumerate(class_names):
            class_probs = probabilities[np.array(labels) == i, 1]
            plt.hist(class_probs, bins=20, alpha=0.7, label=f'{class_name}')
        plt.xlabel('Predicted Probability (Parkinson)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predicted Probabilities')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        correct_mask = np.array(labels) == np.array(predictions)
        incorrect_mask = ~correct_mask
        
        plt.hist(probabilities[correct_mask, 1], bins=20, alpha=0.7, 
                label='Correct', color='green')
        plt.hist(probabilities[incorrect_mask, 1], bins=20, alpha=0.7,
                label='Incorrect', color='red')
        plt.xlabel('Predicted Probability (Parkinson)')
        plt.ylabel('Frequency')
        plt.title('Probability Distribution by Correctness')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'probability_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        results['auc'] = float(roc_auc)
    
    return results


def load_model_for_inference(model_class, checkpoint_path, device='cpu'):
    """
    Carga modelo para inferencia.
    """
    # Crear modelo
    model = model_class()
    
    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Modo evaluación
    model.eval()
    model.to(device)
    
    return model, checkpoint


def count_parameters(model):
    """Cuenta parámetros del modelo."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def get_model_summary(model, input_shapes):
    """
    Genera resumen del modelo.
    
    Args:
        model: Modelo PyTorch
        input_shapes: Lista de shapes de input [(C, H, W), ...]
    """
    from torchsummary import summary
    
    try:
        if len(input_shapes) == 1:
            summary(model, input_shapes[0])
        else:
            # Para modelos con múltiples inputs, usar implementación manual
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print("=" * 50)
            print("Model Summary")
            print("=" * 50)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Non-trainable parameters: {total_params - trainable_params:,}")
            print("=" * 50)
            
    except ImportError:
        # Fallback si torchsummary no está disponible
        params = count_parameters(model)
        print("=" * 50)
        print("Model Summary")
        print("=" * 50)
        print(f"Total parameters: {params['total']:,}")
        print(f"Trainable parameters: {params['trainable']:,}")
        print(f"Non-trainable parameters: {params['non_trainable']:,}")
        print("=" * 50)