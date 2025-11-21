import torch
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import roc_curve, f1_score
from data_loader import create_data_loaders
from architecture import create_model
from utils import load_checkpoint

"""Calibración de threshold para Parkinson.
Genera un archivo JSON con threshold recomendado basado en Youden J y F1.
Uso:
    python -m model.calibrate_threshold --checkpoint model/checkpoints/best_model.pth --metadata data/metadata/metadata_features_final.csv
"""

import argparse

def compute_best_threshold(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    youden = tpr - fpr
    best_youden_idx = np.argmax(youden)
    best_youden_thr = thresholds[best_youden_idx]

    # F1 Parkinson sobre grid
    f1_scores = []
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        f1 = f1_score(labels, preds, pos_label=1)
        f1_scores.append(f1)
    best_f1_idx = int(np.argmax(f1_scores))
    best_f1_thr = thresholds[best_f1_idx]

    return {
        'best_youden_threshold': float(best_youden_thr),
        'best_youden_score': float(youden[best_youden_idx]),
        'best_f1_threshold': float(best_f1_thr),
        'best_f1_score': float(f1_scores[best_f1_idx])
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='model/checkpoints/best_model.pth')
    parser.add_argument('--metadata', type=str, default='data/metadata/metadata_features_final.csv')
    parser.add_argument('--feature_type', type=str, default='both')
    parser.add_argument('--output', type=str, default='model/results/calibration_threshold.json')
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint no encontrado: {args.checkpoint}")
        return
    if not Path(args.metadata).exists():
        print(f"Metadata no encontrada: {args.metadata}")
        return

    # Crear loaders (queremos validation + test juntos para calibrar si es pequeño)
    loaders = create_data_loaders(
        metadata_file=args.metadata,
        batch_size=16,
        test_size=0.2,
        validation_size=0.2,
        feature_type=args.feature_type,
        augment_train=False,
        verbose=False
    )

    val_loader = loaders['val']

    # Cargar modelo
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    # Inferir arquitectura (asumimos híbrido)
    model = create_model(model_type='hybrid', num_classes=2)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            if args.feature_type == 'both':
                mfcc, mel, labels = batch
                outputs = model(mfcc, mel)
            elif args.feature_type == 'mfcc':
                mfcc, labels = batch
                outputs = model(mfcc)
            else:
                mel, labels = batch
                outputs = model(mel)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.squeeze().cpu().numpy())

    metrics = compute_best_threshold(np.array(all_labels), np.array(all_probs))
    metrics['num_samples'] = int(len(all_labels))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Calibración guardada en {args.output}\n{metrics}")

if __name__ == '__main__':
    main()
