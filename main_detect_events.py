import sys
from pathlib import Path
import yaml
import joblib
import numpy as np

# ============================================================
# Configuración de proyecto
# ============================================================

PROJECT_ROOT = Path.cwd()
sys.path.append(str(PROJECT_ROOT))
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

from preprocessing.das_preprocess import (
    read_das_npz,
    downsample_das
)

from detection.event_detector import DASEventDetector


def main():

    # --------------------------------------------------------
    # 1. Cargar configuración
    # --------------------------------------------------------
    if not CONFIG_PATH.exists():
        print(f"Error: No se encuentra el archivo {CONFIG_PATH}")
        return

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # --------------------------------------------------------
    # 2. Lectura de datos
    # --------------------------------------------------------
    RAW_DATA_DIR = PROJECT_ROOT / "data"
    npz_files = list(RAW_DATA_DIR.glob("*.npz"))

    if not npz_files:
        print("No se encontraron archivos .npz")
        return

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_events = []

    # --------------------------------------------------------
    # 3. Procesar archivos DAS
    # --------------------------------------------------------
    for file_path in npz_files:
        print(f"\nProcesando {file_path.name}")

        # --- Lectura ---
        das = read_das_npz(file_path)

        # --- Downsampling ---
        target_fs = cfg["signal"].get("target_fs", 1000)
        das = downsample_das(das, target_fs)

        X_raw = das["strain_data"]     # (T, S)
        t = das["t"]
        fs = das["sampling_freq"]

        # ====================================================
        # 3. DETECCIÓN (UNIFICADO)
        # ====================================================
        detector = DASEventDetector(
            fs=fs,
            fmin=cfg["signal"]["fmin"],
            fmax=cfg["signal"]["fmax"],
            smooth_window_sec=cfg["detection"].get("smooth_sec", 0.5),
            threshold=cfg["detection"].get("threshold", 3.0),
            min_duration_sec=cfg["detection"].get("min_duration_sec", 1.0),
            min_sensors=cfg["detection"].get("min_sensors", 5),
            sigma_2d=cfg["signal"].get("sigma_2d", 1.2),
            clip_percentile=cfg["detection"].get("clip_percentile", 2)
        )
        
        events, E_norm, mask, X_proc = detector.detect(X_raw, time_axis=t)

        for ev in events:
            ev["file"] = file_path.name
            all_events.append(ev)

        # ====================================================
        # 4. GUARDAR PARA NOTEBOOK (ORGANIZADO)
        # ====================================================
        file_out_dir = out_dir / file_path.stem
        file_out_dir.mkdir(parents=True, exist_ok=True)

        np.save(file_out_dir / "energy.npy", E_norm)
        np.save(file_out_dir / "mask.npy", mask)
        joblib.dump(events, file_out_dir / "events.pkl")

        print(f"  → {len(events)} eventos detectados")

    # --------------------------------------------------------
    # 7. Guardar eventos globales (Resumen)
    # --------------------------------------------------------
    joblib.dump(all_events, out_dir / "all_events.pkl")

    print("\n✔ Proceso completado")
    print(f"✔ Total eventos detectados: {len(all_events)}")
    print(f"✔ Resultados organizados en {out_dir}")


if __name__ == "__main__":
    main()
