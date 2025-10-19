
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Imports locales
from architecture import create_model
from data_loader import create_data_loaders
from utils import EarlyStopping, MetricsTracker, save_checkpoint, load_checkpoint


class NeuroVoiceTrainer:
    """
    Entrenador principal para el modelo NeuroVoice.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Crear directorios
        self.setup_directories()
        
        # Inicializar componentes
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Métricas y tracking
        self.metrics_tracker = MetricsTracker()
        self.early_stopping = EarlyStopping(
            patience=config['early_stopping_patience'],
            min_delta=config['early_stopping_min_delta']
        )
        
        # Mejores métricas
        self.best_val_acc = 0.0
        self.best_val_auc = 0.0
        self.best_epoch = 0
    
    def setup_directories(self):
        """Crea directorios necesarios."""
        dirs = [
            'model/checkpoints',
            'model/results', 
            'model/logs',
            'model/plots'
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def setup_model_and_data(self):
        """Inicializa modelo, datos y componentes de entrenamiento."""
        # Crear data loaders
        data_loaders = create_data_loaders(
            metadata_file=self.config['metadata_file'],
            batch_size=self.config['batch_size'],
            test_size=self.config['test_size'],
            validation_size=self.config['validation_size'],
            feature_type=self.config['feature_type'],
            augment_train=self.config['augment_train'],
            random_state=self.config['random_state'],
            num_workers=self.config['num_workers'],
            verbose=False
        )
        
        self.train_loader = data_loaders['train']
        self.val_loader = data_loaders['val']
        self.test_loader = data_loaders['test']
        self.datasets = data_loaders['datasets']
        
        # Crear modelo
        self.model = create_model(
            model_type=self.config['model_type'],
            num_classes=self.config['num_classes'],
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)
        
        # Loss function con class weights si está configurado
        if self.config['use_class_weights']:
            class_weights = self.datasets['train'].get_class_weights().to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['weight_decay']
            )
        
        # Scheduler
        if self.config['scheduler'] == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif self.config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['num_epochs'],
                eta_min=1e-6
            )
    
    def train_epoch(self, epoch):
        """Entrena una época."""
        self.model.train()
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
        
        for batch_idx, batch_data in enumerate(pbar):
            if self.config['feature_type'] == 'both':
                mfcc, mel, labels = batch_data
                mfcc, mel = mfcc.to(self.device), mel.to(self.device)
                labels = labels.to(self.device).squeeze()
                
                # Forward pass
                outputs = self.model(mfcc, mel)
            else:
                features, labels = batch_data
                features = features.to(self.device)
                labels = labels.to(self.device).squeeze()
                
                outputs = self.model(features)
            
            # Calcular loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config['gradient_clipping'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clipping']
                )
            
            self.optimizer.step()
            
            # Estadísticas
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Para métricas
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Actualizar progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{100 * correct_predictions / total_samples:.2f}%"
            })
        
        # Métricas de la época
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct_predictions / total_samples
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def validate_epoch(self, epoch):
        """Valida una época."""
        self.model.eval()
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Validating"):
                if self.config['feature_type'] == 'both':
                    mfcc, mel, labels = batch_data
                    mfcc, mel = mfcc.to(self.device), mel.to(self.device)
                    labels = labels.to(self.device).squeeze()
                    
                    outputs = self.model(mfcc, mel)
                else:
                    features, labels = batch_data
                    features = features.to(self.device)
                    labels = labels.to(self.device).squeeze()
                    
                    outputs = self.model(features)
                
                loss = self.criterion(outputs, labels)
                
                # Estadísticas
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                # Para métricas avanzadas
                probabilities = torch.softmax(outputs, dim=1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Métricas de validación
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct_predictions / total_samples
        
        # AUC Score
        try:
            epoch_auc = roc_auc_score(all_labels, np.array(all_probabilities)[:, 1])
        except:
            epoch_auc = 0.0
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'auc': epoch_auc,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
    
    def train(self):
        """Función principal de entrenamiento."""
        # Setup
        self.setup_model_and_data()
        
        # Guardar configuración
        config_path = Path('model/logs') / f'config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(self.config['num_epochs']):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate_epoch(epoch)
            
            # Update scheduler
            if self.config['scheduler'] == 'reduce_on_plateau':
                self.scheduler.step(val_metrics['accuracy'])
            elif self.config['scheduler'] == 'cosine':
                self.scheduler.step()
            
            # Track metrics
            self.metrics_tracker.update(epoch, train_metrics, val_metrics)
            
            # Epoch time
            epoch_time = time.time() - epoch_start
            
            # Logging
            print(f"Epoch {epoch+1}/{self.config['num_epochs']} - Train: {train_metrics['accuracy']:.1f}%, Val: {val_metrics['accuracy']:.1f}% (AUC: {val_metrics['auc']:.3f})")
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_auc = val_metrics['auc']
                self.best_epoch = epoch
                
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    filepath='model/checkpoints/best_model.pth'
                )
                print(f"NEW BEST: {self.best_val_acc:.1f}%")
            
            # Early stopping
            if self.early_stopping(val_metrics['accuracy']):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    filepath=f'model/checkpoints/checkpoint_epoch_{epoch+1}.pth'
                )
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\nTraining completed - Best: {self.best_val_acc:.1f}% (epoch {self.best_epoch+1})")
        
        # Final evaluation
        self.final_evaluation()
        
        # Save training history
        self.metrics_tracker.save_history('model/results/training_history.json')
        
        # Generate plots
        self.generate_plots()
    
    def final_evaluation(self):
        """Evaluación final en test set."""
        # Load best model
        checkpoint = load_checkpoint('model/checkpoints/best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate on test set
        test_metrics = self.validate_epoch(-1)  # Usar validate_epoch para test
        
        print(f"\nFINAL RESULTS:")
        print(f"Test Accuracy: {test_metrics['accuracy']:.1f}% | AUC: {test_metrics['auc']:.3f}")
        
        # Classification report
        class_names = ['Healthy', 'Parkinson']
        report = classification_report(
            test_metrics['labels'],
            test_metrics['predictions'],
            target_names=class_names,
            output_dict=True
        )
        
        print(f"Classification Report:")
        print(classification_report(
            test_metrics['labels'],
            test_metrics['predictions'],
            target_names=class_names
        ))
        
        # Save results
        results = {
            'test_accuracy': float(test_metrics['accuracy']),
            'test_auc': float(test_metrics['auc']),
            'classification_report': report,
            'best_epoch': int(self.best_epoch),
            'best_val_accuracy': float(self.best_val_acc),
            'best_val_auc': float(self.best_val_auc)
        }
        
        results_path = 'model/results/final_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
    
    def generate_plots(self):
        """Genera gráficos de entrenamiento."""
        history = self.metrics_tracker.get_history()
        
        # Training curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(history['train_accuracy'], label='Train Accuracy', color='blue')
        axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC curve
        axes[1, 0].plot(history['val_auc'], label='Validation AUC', color='green')
        axes[1, 0].set_title('Validation AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate (if available)
        if hasattr(self.scheduler, 'get_last_lr'):
            lr_history = []
            # This would need to be tracked during training
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('model/plots/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()




def main():
    parser = argparse.ArgumentParser(description='Train NeuroVoice CNN')
    parser.add_argument('--config', type=str, default='model/config.json',
                       help='Path to config JSON file')
    parser.add_argument('--metadata', type=str, 
                       help='Path to metadata CSV file')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--model_type', type=str, 
                       choices=['hybrid', 'mel_only', 'mfcc_only'],
                       help='Model architecture type')
    
    args = parser.parse_args()
    
    # Load config from __init__.py system
    import sys
    sys.path.append('.')
    from model import NeuroVoiceConfig
    
    # Load configuration
    if Path(args.config).exists():
        config_manager = NeuroVoiceConfig(args.config, verbose=True)
        config = config_manager.get_training_config()
        print(f"Configuration loaded from {args.config}")
    else:
        print(f"Config file not found: {args.config}")
        print("Create config.json by running: python model/__init__.py")
        return
    
    # Override with command line arguments if provided
    if args.metadata:
        config['metadata_file'] = args.metadata
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    if args.model_type:
        config['model_type'] = args.model_type
    
    # Set random seeds
    torch.manual_seed(config['random_state'])
    np.random.seed(config['random_state'])
    
    print("NeuroVoice Training Started")
    print("=" * 40)
    
    # Create trainer and start training
    trainer = NeuroVoiceTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()