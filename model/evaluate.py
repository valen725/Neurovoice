

import torch
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve
)

# Imports locales
from architecture import create_model
from data_loader import create_data_loaders
from utils import ModelEvaluator, load_checkpoint, analyze_model_predictions


class NeuroVoiceEvaluator:
    """
    Evaluador completo para modelos NeuroVoice.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Crear directorios para resultados
        self.results_dir = Path('model/evaluation_results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar componentes
        self.model = None
        self.test_loader = None
        self.class_names = ['Healthy', 'Parkinson']
    
    def load_model(self, checkpoint_path):
        """Carga el modelo desde checkpoint."""
        print(f"Loading model from {checkpoint_path}")
        
        # Crear modelo
        self.model = create_model(
            model_type=self.config['model_type'],
            num_classes=self.config['num_classes'],
            dropout_rate=self.config.get('dropout_rate', 0.3)
        )
        
        # Cargar checkpoint
        checkpoint = load_checkpoint(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f" Model loaded successfully")
        print(f"   Epoch: {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            print(f"   Val Accuracy: {metrics.get('accuracy', 'N/A'):.2f}%")
            print(f"   Val AUC: {metrics.get('auc', 'N/A'):.4f}")
        
        return checkpoint
    
    def load_data(self):
        """Carga los datos de evaluación."""
        print("Loading evaluation data")
        
        data_loaders = create_data_loaders(
            metadata_file=self.config['metadata_file'],
            batch_size=self.config['batch_size'],
            test_size=self.config['test_size'],
            validation_size=self.config['validation_size'],
            feature_type=self.config['feature_type'],
            augment_train=False,  # No augmentation for evaluation
            random_state=self.config['random_state'],
            num_workers=self.config['num_workers']
        )
        
        self.test_loader = data_loaders['test']
        self.val_loader = data_loaders['val']
        self.train_loader = data_loaders['train']
        
        print(f"Data loaded successfully")
        print(f"   Test samples: {len(data_loaders['datasets']['test'])}")
        print(f"   Val samples: {len(data_loaders['datasets']['val'])}")
        print(f"   Train samples: {len(data_loaders['datasets']['train'])}")
    
    def evaluate_model(self, data_loader, dataset_name="Test"):
        """Evalúa el modelo en un dataset."""
        print(f"Evaluating on {dataset_name} set")
        
        evaluator = ModelEvaluator(self.model, self.device)
        results = evaluator.evaluate_dataset(data_loader, self.class_names)
        
        print(f"{dataset_name} Results:")
        print(f"   Accuracy: {results['accuracy']:.2f}%")
        print(f"   AUC: {results['auc']:.4f}")
        print(f"   Loss: {results['loss']:.4f}")
        
        return results
    
    def generate_detailed_report(self, test_results, val_results=None):
        """Genera reporte detallado de evaluación."""

        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_config': self.config,
            'test_results': {
                'accuracy': float(test_results['accuracy']),
                'auc': float(test_results['auc']),
                'loss': float(test_results['loss']),
                'classification_report': test_results['classification_report']
            }
        }
        
        if val_results:
            report['validation_results'] = {
                'accuracy': float(val_results['accuracy']),
                'auc': float(val_results['auc']),
                'loss': float(val_results['loss']),
                'classification_report': val_results['classification_report']
            }
        
        # Guardar reporte
        report_path = self.results_dir / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f" Detailed report saved to {report_path}")
        return report
    
    def generate_visualizations(self, results, dataset_name="Test"):
        """Genera visualizaciones de los resultados."""

        
        labels = results['labels']
        predictions = results['predictions']
        probabilities = np.array(results['probabilities'])
        
        # Crear directorio para plots
        plots_dir = self.results_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Matriz de Confusión
        self._plot_confusion_matrix(labels, predictions, plots_dir, dataset_name)
        
        # 2. Curva ROC
        self._plot_roc_curve(labels, probabilities, plots_dir, dataset_name)
        
        # 3. Curva Precision-Recall
        self._plot_precision_recall_curve(labels, probabilities, plots_dir, dataset_name)
        
        # 4. Distribución de Probabilidades
        self._plot_probability_distributions(labels, predictions, probabilities, plots_dir, dataset_name)
        
        # 5. Métricas por Clase
        self._plot_class_metrics(results['classification_report'], plots_dir, dataset_name)
        
        print(f" Visualizations saved to {plots_dir}")
    
    def _plot_confusion_matrix(self, labels, predictions, save_dir, dataset_name):
        """Genera matriz de confusión."""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {dataset_name} Set')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Agregar porcentajes
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                total = cm.sum()
                percentage = cm[i, j] / total * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'confusion_matrix_{dataset_name.lower()}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self, labels, probabilities, save_dir, dataset_name):
        """Genera curva ROC."""
        fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {dataset_name} Set')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / f'roc_curve_{dataset_name.lower()}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curve(self, labels, probabilities, save_dir, dataset_name):
        """Genera curva Precision-Recall."""
        precision, recall, _ = precision_recall_curve(labels, probabilities[:, 1])
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {dataset_name} Set')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / f'precision_recall_curve_{dataset_name.lower()}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_probability_distributions(self, labels, predictions, probabilities, save_dir, dataset_name):
        """Genera distribución de probabilidades."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distribución por clase real
        axes[0, 0].hist(probabilities[np.array(labels) == 0, 1], bins=20, alpha=0.7, 
                       label='Healthy', color='green', density=True)
        axes[0, 0].hist(probabilities[np.array(labels) == 1, 1], bins=20, alpha=0.7,
                       label='Parkinson', color='red', density=True)
        axes[0, 0].set_xlabel('Predicted Probability (Parkinson)')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Probability Distribution by True Class')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribución por predicción correcta/incorrecta
        correct_mask = np.array(labels) == np.array(predictions)
        axes[0, 1].hist(probabilities[correct_mask, 1], bins=20, alpha=0.7,
                       label='Correct', color='blue', density=True)
        axes[0, 1].hist(probabilities[~correct_mask, 1], bins=20, alpha=0.7,
                       label='Incorrect', color='orange', density=True)
        axes[0, 1].set_xlabel('Predicted Probability (Parkinson)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Probability Distribution by Prediction Correctness')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Box plot por clase
        prob_data = [probabilities[np.array(labels) == i, 1] for i in range(2)]
        axes[1, 0].boxplot(prob_data, labels=self.class_names)
        axes[1, 0].set_ylabel('Predicted Probability (Parkinson)')
        axes[1, 0].set_title('Probability Distribution by True Class (Box Plot)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Scatter plot de confianza
        confidence = np.max(probabilities, axis=1)
        correct_colors = ['green' if c else 'red' for c in correct_mask]
        axes[1, 1].scatter(range(len(confidence)), confidence, c=correct_colors, alpha=0.6)
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Prediction Confidence')
        axes[1, 1].set_title('Prediction Confidence by Sample (Green=Correct, Red=Incorrect)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'probability_distributions_{dataset_name.lower()}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_class_metrics(self, classification_report, save_dir, dataset_name):
        """Genera gráfico de métricas por clase."""
        # Extraer métricas por clase
        metrics_data = []
        for class_name in self.class_names:
            if class_name.lower() in classification_report:
                class_metrics = classification_report[class_name.lower()]
                metrics_data.append({
                    'Class': class_name,
                    'Precision': class_metrics['precision'],
                    'Recall': class_metrics['recall'],
                    'F1-Score': class_metrics['f1-score']
                })
        
        if not metrics_data:
            return
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Plot de barras
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(self.class_names))
        width = 0.25
        
        ax.bar(x - width, df_metrics['Precision'], width, label='Precision', alpha=0.8)
        ax.bar(x, df_metrics['Recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + width, df_metrics['F1-Score'], width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title(f'Classification Metrics by Class - {dataset_name} Set')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        # Agregar valores en las barras
        for i, (precision, recall, f1) in enumerate(zip(df_metrics['Precision'], 
                                                        df_metrics['Recall'],
                                                        df_metrics['F1-Score'])):
            ax.text(i - width, precision + 0.02, f'{precision:.3f}', ha='center', va='bottom')
            ax.text(i, recall + 0.02, f'{recall:.3f}', ha='center', va='bottom')
            ax.text(i + width, f1 + 0.02, f'{f1:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'class_metrics_{dataset_name.lower()}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_with_baseline(self, results):
        """Compara resultados con baseline (modelo simple)."""

        
        labels = results['labels']
        
        # Baseline: predicción de clase mayoritaria
        majority_class = max(set(labels), key=labels.count)
        baseline_predictions = [majority_class] * len(labels)
        baseline_accuracy = sum(1 for true, pred in zip(labels, baseline_predictions) 
                              if true == pred) / len(labels) * 100
        
        model_accuracy = results['accuracy']
        improvement = model_accuracy - baseline_accuracy
        
        print(f"   Baseline accuracy (majority class): {baseline_accuracy:.2f}%")
        print(f"   Model accuracy: {model_accuracy:.2f}%")
        print(f"   Improvement: {improvement:.2f} percentage points")
        
        return {
            'baseline_accuracy': baseline_accuracy,
            'model_accuracy': model_accuracy,
            'improvement': improvement
        }
    
    def run_complete_evaluation(self, checkpoint_path):
        """Ejecuta evaluación completa."""
        print("Starting complete evaluation")
        print("=" * 60)
        
        # Cargar modelo y datos
        checkpoint = self.load_model(checkpoint_path)
        self.load_data()
        
        # Evaluar en test set
        test_results = self.evaluate_model(self.test_loader, "Test")
        
        # Evaluar en validation set para comparación
        val_results = self.evaluate_model(self.val_loader, "Validation")
        
        # Generar visualizaciones
        self.generate_visualizations(test_results, "Test")
        self.generate_visualizations(val_results, "Validation")
        
        # Comparar con baseline
        baseline_comparison = self.compare_with_baseline(test_results)
        
        # Generar reporte detallado
        report = self.generate_detailed_report(test_results, val_results)
        report['baseline_comparison'] = baseline_comparison
        
        # Guardar reporte actualizado
        report_path = self.results_dir / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        # Resumen final
        print("\n" + "=" * 60)
        print(" EVALUATION COMPLETE")
        print("=" * 60)
        print(f" Test Accuracy: {test_results['accuracy']:.2f}%")
        print(f" Test AUC: {test_results['auc']:.4f}")
        print(f" Validation Accuracy: {val_results['accuracy']:.2f}%")
        print(f" Validation AUC: {val_results['auc']:.4f}")
        print(f" Improvement over baseline: {baseline_comparison['improvement']:.2f}pp")
        print(f"Results saved to: {self.results_dir}")
        print("=" * 60)
        
        return report


def get_default_config():
    """Configuración por defecto para evaluación."""
    return {
        'metadata_file': 'data/metadata/metadata_features_final.csv',
        'batch_size': 32,
        'test_size': 0.2,
        'validation_size': 0.2,
        'feature_type': 'both',
        'model_type': 'hybrid',
        'num_classes': 2,
        'dropout_rate': 0.3,
        'num_workers': 4,
        'random_state': 42
    }


def find_best_model():
    """Busca el mejor modelo en ubicaciones comunes."""
    possible_paths = [
        "model/checkpoints/best_model.pth",
        "checkpoints/best_model.pth", 
        "best_model.pth"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return path
    return None

def main():
    parser = argparse.ArgumentParser(description='Evaluate NeuroVoice CNN')
    parser.add_argument('--checkpoint', type=str, 
                       help='Path to model checkpoint (default: finds best_model.pth automatically)')
    parser.add_argument('--config', type=str, 
                       help='Path to config JSON file')
    parser.add_argument('--metadata', type=str,
                       default='data/metadata/metadata_features_final.csv',
                       help='Path to metadata CSV file')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--model_type', type=str, default='hybrid',
                       choices=['hybrid', 'mel_only', 'mfcc_only'],
                       help='Model architecture type')
    
    args = parser.parse_args()
    
    # Si no se especifica checkpoint, buscar best_model.pth automáticamente
    if not args.checkpoint:
        checkpoint_path = find_best_model()
        if checkpoint_path:
            print(f"Using automatically detected model: {checkpoint_path}")
            args.checkpoint = checkpoint_path
        else:
            print("Error: best_model.pth not found in common locations.")
            print("Specify the path manually with --checkpoint")
            print("Searched locations:")
            print("   - model/checkpoints/best_model.pth")
            print("   - checkpoints/best_model.pth") 
            print("   - best_model.pth")
            return
    
    # Verificar que el checkpoint existe
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    # Cargar configuración
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Override con argumentos de línea de comandos
    if args.metadata:
        config['metadata_file'] = args.metadata
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.model_type:
        config['model_type'] = args.model_type
    
    print("NeuroVoice Model Evaluation")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print("=" * 50)
    
    # Crear evaluador y ejecutar evaluación
    evaluator = NeuroVoiceEvaluator(config)
    report = evaluator.run_complete_evaluation(args.checkpoint)
    
    print("\nEVALUATION COMPLETED")
    print("=" * 50)
    print("Results saved to:")
    print("   model/evaluation_results/evaluation_report.json")
    print("   model/evaluation_results/plots/")
    print("=" * 50)


if __name__ == "__main__":
    main()