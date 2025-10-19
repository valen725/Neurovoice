import argparse
from pathlib import Path
import pandas as pd, numpy as np
import librosa, soundfile as sf
from tqdm import tqdm

def simple_vad(y, top_db=25):
    """Elimina silencios prolongados al inicio o final."""
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return y
    return np.concatenate([y[s:e] for s, e in intervals])

def peak_normalize(y, peak=0.99):
    """Normaliza la amplitud al pico máximo."""
    m = np.max(np.abs(y)) + 1e-9
    return (y / m) * peak

def pad_or_crop(y, target_len):
    """Ajusta el audio al tamaño exacto (recorta o rellena)."""
    n = len(y)
    if n > target_len:
        start = (n - target_len) // 2
        return y[start:start + target_len]
    if n < target_len:
        pad_left = (target_len - n) // 2
        pad_right = target_len - n - pad_left
        return np.pad(y, (pad_left, pad_right), mode="constant")
    return y

def process_one(in_path: Path, out_path: Path, target_sr=16000, target_sec=3.0, vad_db=25):
    """Procesa un solo archivo: remuestrea, limpia y guarda."""
    y, sr = librosa.load(in_path, sr=None, mono=True)
    # 1. Remuestreo a 16kHz
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
        sr = target_sr
    # 2. Eliminar silencios
    y = simple_vad(y, top_db=vad_db)
    # 3. Normalizar volumen
    y = peak_normalize(y)
    # 4. Fijar duración
    target_len = int(target_sr * target_sec)
    y = pad_or_crop(y, target_len)
    # 5. Guardar archivo
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, y, sr, subtype="PCM_16")
    return len(y) / sr, sr

def main():
    ap = argparse.ArgumentParser(description="Remuestrea a 16kHz y fija duración 3s.")
    ap.add_argument("--inmeta", default="data/metadata/metadata_master.csv")
    ap.add_argument("--outmeta", default="data/metadata/metadata_processed.csv")
    ap.add_argument("--outroot", default="data/processed")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--vad_db", type=int, default=25)
    args = ap.parse_args()

    df = pd.read_csv(args.inmeta)
    rows = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Procesando"):
        if not bool(r.get("valid", True)):
            continue
        src = Path(r["filepath"])
        label = r["label"].strip().lower()
        fname = r["filename"]
        out_fp = Path(args.outroot) / label / fname

        try:
            dur, sr = process_one(src, out_fp, args.sr, args.duration, args.vad_db)
            rows.append({
                "src_filepath": src.as_posix(),
                "filepath": out_fp.as_posix(),
                "label": label,
                "filename": fname,
                "duration_sec": round(dur, 6),
                "sample_rate": sr,
                "valid": True
            })
        except Exception as e:
            rows.append({
                "src_filepath": src.as_posix(),
                "filepath": out_fp.as_posix(),
                "label": label,
                "filename": fname,
                "duration_sec": np.nan,
                "sample_rate": np.nan,
                "valid": False,
                "error": str(e)
            })

    outmeta = Path(args.outmeta)
    outmeta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outmeta, index=False)
    print(f"\n✅ Procesamiento completado. CSV: {outmeta}\nWAVs guardados en: {args.outroot}")

if __name__ == "__main__":
    main()
