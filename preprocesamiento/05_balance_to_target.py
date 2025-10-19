import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import shutil
import random

# ------------------ Utils de audio ------------------
def pad_or_crop(y, target_len):
    n = len(y)
    if n == target_len:
        return y
    if n > target_len:
        start = (n - target_len) // 2
        return y[start:start+target_len]
    pad_left = (target_len - n) // 2
    pad_right = target_len - n - pad_left
    return np.pad(y, (pad_left, pad_right), mode="constant")

def rms(x):
    return np.sqrt(np.mean(np.square(x)) + 1e-12)

def add_noise_snr(y, snr_db=15.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    sig_rms = rms(y)
    snr_lin = 10 ** (snr_db / 10.0)
    noise_rms = sig_rms / np.sqrt(snr_lin)
    noise = rng.normal(0.0, noise_rms, size=y.shape)
    z = y + noise
    peak = np.max(np.abs(z)) + 1e-9
    if peak > 1.0:
        z = z / peak * 0.99
    return z

def do_pitch(y, sr, steps):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

def do_stretch(y, rate, target_len):
    z = librosa.effects.time_stretch(y, rate=rate)
    return pad_or_crop(z, target_len)

def save_wav(path, y, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, sr, subtype="PCM_16")

# ------------------ Lógica principal ------------------
def main():
    ap = argparse.ArgumentParser(description="Balancea a un objetivo por clase (100/100) creando solo los aumentos necesarios.")
    ap.add_argument("--inmeta", default="data/metadata/metadata_processed.csv", help="CSV con audios ya procesados (16kHz/3s)")
    ap.add_argument("--outmeta", default="data/metadata/metadata_final_for_training.csv", help="CSV final con 100/100")
    ap.add_argument("--finalroot", default="data/final_for_training", help="Carpeta final donde quedarán exactamente 100 por clase")
    ap.add_argument("--target_per_class", type=int, default=100, help="Objetivo final por clase")
    ap.add_argument("--sr", type=int, default=16000, help="Frecuencia esperada")
    ap.add_argument("--duration", type=float, default=3.0, help="Duración esperada (s)")
    ap.add_argument("--seed", type=int, default=42, help="Seed para reproducibilidad")
    # parámetros de augmentations disponibles
    ap.add_argument("--snr_list", nargs="*", type=float, default=[15.0, 20.0], help="SNRs posibles para ruido")
    ap.add_argument("--pitch_list", nargs="*", type=float, default=[+1.5, -1.5], help="Semitonos para pitch shift")
    ap.add_argument("--stretch_list", nargs="*", type=float, default=[0.95, 1.05], help="Factores de time-stretch")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    df = pd.read_csv(args.inmeta)
    df = df[df["valid"] == True].copy()
    assert set(df["label"].unique()) <= {"healthy", "parkinson"}, "Labels deben ser healthy/parkinson"

    finalroot = Path(args.finalroot)
    (finalroot / "healthy").mkdir(parents=True, exist_ok=True)
    (finalroot / "parkinson").mkdir(parents=True, exist_ok=True)

    target_len = int(args.sr * args.duration)

    # 1) Copiar todos los audios procesados a la carpeta final
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Copiando base procesada"):
        src = Path(r["filepath"])
        label = r["label"].strip().lower()
        dst = finalroot / label / Path(r["filename"]).name
        # evita copia redundante si ya existe el mismo archivo
        if not dst.exists():
            shutil.copy2(src, dst)
        rows.append({
            "filepath": dst.as_posix(),
            "label": label,
            "filename": dst.name,
            "source": "processed_base",
            "valid": True,
            "sample_rate": args.sr,
            "duration_sec": args.duration
        })

    # 2) Chequear cuántos hay por clase y cuántos faltan hasta target
    current = pd.DataFrame(rows).groupby("label")["filename"].count().to_dict()
    need = {lab: max(0, args.target_per_class - current.get(lab, 0)) for lab in ["healthy", "parkinson"]}
    print(f"Conteo actual por clase: {current} | Faltan: {need}")

    # 3) Si falta, generar aumentos necesarios
    if any(n > 0 for n in need.values()):
        # Para generar, usamos los archivos base como fuente
        base_by_label = {lab: df[df["label"] == lab]["filepath"].tolist() for lab in ["healthy", "parkinson"]}

        # Definimos una “ruleta” de transformaciones disponibles
        # Cada entrada es (nombre, función-aplicación, sufijo-nombre)
        def make_aug_ops():
            ops = []
            # Ruido con SNRs
            for snr in args.snr_list:
                ops.append(("noise", lambda y, sr: add_noise_snr(y, snr_db=snr, rng=rng), f"noise{int(snr)}dB"))
            # Pitch
            for st in args.pitch_list:
                ops.append(("pitch", lambda y, sr, st=st: do_pitch(y, sr, st), f"pitch{st:+.1f}st"))
            # Stretch
            for rt in args.stretch_list:
                ops.append(("stretch", lambda y, sr, rt=rt: do_stretch(y, rate=rt, target_len=target_len), f"stretch{rt:.2f}x"))
            return ops

        aug_ops = make_aug_ops()

        for lab in ["healthy", "parkinson"]:
            to_make = need[lab]
            if to_make == 0:
                continue
            candidates = base_by_label[lab]
            if len(candidates) == 0:
                raise RuntimeError(f"No hay base para {lab} y se necesitan {to_make} aumentos.")

            i = 0
            while i < to_make:
                # elegimos una fuente al azar y una transformación al azar
                src_fp = Path(random.choice(candidates))
                y, sr = librosa.load(src_fp, sr=None, mono=True)
                if sr != args.sr:
                    y = librosa.resample(y, orig_sr=sr, target_sr=args.sr, res_type="scipy")
                    sr = args.sr
                y = pad_or_crop(y, target_len)

                op_name, op_fn, op_suffix = random.choice(aug_ops)
                y_aug = op_fn(y, sr)
                y_aug = pad_or_crop(y_aug, target_len)

                base_name = src_fp.stem
                out_name = f"{base_name}__aug-{op_suffix}__{i:04d}.wav"
                out_fp = finalroot / lab / out_name
                # evitar colisiones raras
                if out_fp.exists():
                    i += 1
                    continue
                save_wav(out_fp, y_aug, sr)
                rows.append({
                    "filepath": out_fp.as_posix(),
                    "label": lab,
                    "filename": out_name,
                    "source": f"aug_{op_name}",
                    "valid": True,
                    "sample_rate": sr,
                    "duration_sec": round(len(y_aug)/sr, 6)
                })
                i += 1

    # 4) Ensamblar dataframe final y (si sobra) recortar a target exacto
    final_df = pd.DataFrame(rows)
    # Si por alguna razón hubiese más de target, muestreamos de forma reproducible
    out_rows = []
    for lab in ["healthy", "parkinson"]:
        sub = final_df[final_df["label"] == lab].copy()
        if len(sub) > args.target_per_class:
            sub = sub.sample(n=args.target_per_class, random_state=args.seed)
        out_rows.append(sub)
    final_df = pd.concat(out_rows, axis=0).sort_values(["label", "filename"]).reset_index(drop=True)

    # 5) Guardar CSV final
    outmeta = Path(args.outmeta)
    outmeta.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(outmeta, index=False)
    print("\n✅ Listo: dataset final balanceado")
    print(f"- Carpeta final: {finalroot.resolve()}")
    print(f"- CSV final:     {outmeta.resolve()}")
    print(final_df.groupby('label')["filename"].count().to_dict())

if __name__ == "__main__":
    main()
