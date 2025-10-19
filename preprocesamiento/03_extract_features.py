import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

def extract_mfcc(y, sr, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return librosa.util.normalize(mfcc)

def extract_mel(y, sr, n_fft=1024, hop_length=512, n_mels=128):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                       hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    return librosa.util.normalize(S_db)

def main():
    ap = argparse.ArgumentParser(description="Extrae MFCC y Mel-spectrogramas")
    ap.add_argument("--inmeta", default="data/metadata/metadata_processed.csv")
    ap.add_argument("--outmeta", default="data/metadata/metadata_features.csv")
    ap.add_argument("--outdir", default="data/features")
    args = ap.parse_args()

    df = pd.read_csv(args.inmeta)
    features = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Extrayendo features"):
        if not bool(r.get("valid", True)):
            continue
        path = Path(r["filepath"])
        label = r["label"].strip().lower()
        try:
            y, sr = librosa.load(path, sr=None, mono=True)
            mfcc = extract_mfcc(y, sr)
            mel = extract_mel(y, sr)

            # nombres de salida
            fname_base = Path(r["filename"]).stem
            out_mfcc = Path(args.outdir) / "mfcc" / f"{fname_base}.npy"
            out_mel  = Path(args.outdir) / "mel"  / f"{fname_base}.npy"

            out_mfcc.parent.mkdir(parents=True, exist_ok=True)
            out_mel.parent.mkdir(parents=True, exist_ok=True)

            np.save(out_mfcc, mfcc)
            np.save(out_mel, mel)

            features.append({
                "filename": r["filename"],
                "label": label,
                "mfcc_path": out_mfcc.as_posix(),
                "mel_path": out_mel.as_posix(),
                "n_mfcc": mfcc.shape,
                "n_mel": mel.shape
            })

        except Exception as e:
            features.append({
                "filename": r["filename"],
                "label": label,
                "mfcc_path": None,
                "mel_path": None,
                "n_mfcc": None,
                "n_mel": None,
                "error": str(e)
            })

    outmeta = Path(args.outmeta)
    pd.DataFrame(features).to_csv(outmeta, index=False)
    print(f"\n✅ Features generados. CSV: {outmeta}")

if __name__ == "__main__":
    main()
