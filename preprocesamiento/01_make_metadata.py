import argparse
from pathlib import Path
import pandas as pd, numpy as np
import soundfile as sf, librosa
from tqdm import tqdm
import sys
from datetime import datetime

def read_audio_info(path):
    """Devuelve (duracion_sec, sample_rate) usando solo soundfile.
    Si soundfile abre el encabezado, consideramos el archivo válido.
    """
    try:
        with sf.SoundFile(path) as f:
            sr = f.samplerate
            frames = len(f)
            dur = frames / float(sr) if sr > 0 else 0.0
        # Considera válido si sr>0 y dur>0 (sin verificación extra con librosa)
        if sr and dur > 0:
            return float(dur), int(sr)
        else:
            return None, None
    except Exception as e:
        print(f"[WARN] No se pudo leer {path}: {e}", file=sys.stderr)
        return None, None


def main():
    ap = argparse.ArgumentParser(description="Genera metadata_master.csv desde data/raw/")
    ap.add_argument("--root", default="data/raw", help="Carpeta con healthy/ y parkinson/")
    ap.add_argument("--outmeta", default="data/metadata/metadata_master.csv", help="Salida CSV")
    ap.add_argument("--reportdir", default="data/metadata/reports", help="Carpeta de reportes")
    args = ap.parse_args()

    root = Path(args.root)
    outmeta = Path(args.outmeta)
    reportdir = Path(args.reportdir)

    classes = ["healthy", "parkinson"]
    rows = []

    for label in classes:
        class_dir = root / label
        class_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(class_dir.glob("*.wav"))
        if len(files) == 0:
            print(f"[WARN] No se encontraron .wav en {class_dir}", file=sys.stderr)
        for f in tqdm(files, desc=f"Procesando {label}", unit="file"):
            dur, sr = read_audio_info(f)
            rows.append({
                "filepath": f.as_posix(),
                "label": label,
                "filename": f.name,
                "duration_sec": np.nan if dur is None else round(dur, 6),
                "sample_rate": np.nan if sr is None else int(sr),
                "valid": False if dur is None else True
            })

    df = pd.DataFrame(rows).sort_values(["label", "filename"]).reset_index(drop=True)

    # Crear carpetas de salida
    outmeta.parent.mkdir(parents=True, exist_ok=True)
    reportdir.mkdir(parents=True, exist_ok=True)

    # Guardar CSV
    df.to_csv(outmeta, index=False)

    # Resumen y reporte
    total = len(df)
    by_label = df.groupby("label")["filename"].count().to_dict()
    valid_count = int(df["valid"].sum())
    invalid = df[~df["valid"]]

    durs = df.loc[df["valid"], "duration_sec"].astype(float)
    stats = {}
    if not durs.empty:
        stats = {
            "min": float(durs.min()),
            "max": float(durs.max()),
            "mean": float(durs.mean()),
            "median": float(durs.median()),
        }

    report_path = reportdir / "metadata_report.txt"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(f"Reporte de Metadata — {datetime.now()}\n")
        fh.write("="*60 + "\n")
        fh.write(f"Root: {root.resolve()}\n")
        fh.write(f"CSV:  {outmeta.resolve()}\n\n")
        fh.write(f"Total de archivos: {total}\n")
        fh.write(f"Archivos válidos:  {valid_count}\n")
        fh.write(f"Archivos inválidos: {total - valid_count}\n")
        fh.write(f"Distribución por clase: {by_label}\n\n")
        if stats:
            fh.write("Duración (s) — válidos:\n")
            fh.write(f"  min:    {stats['min']:.4f}\n")
            fh.write(f"  max:    {stats['max']:.4f}\n")
            fh.write(f"  mean:   {stats['mean']:.4f}\n")
            fh.write(f"  median: {stats['median']:.4f}\n\n")
        if len(invalid) > 0:
            fh.write("Archivos inválidos:\n")
            for p in invalid["filepath"].tolist():
                fh.write(f"  - {p}\n")

    print("\n✅ Listo:")
    print(f"- CSV:     {outmeta}")
    print(f"- Reporte: {report_path}")

if __name__ == "__main__":
    main()
