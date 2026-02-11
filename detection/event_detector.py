# detection/event_detector.py

import numpy as np
from scipy.ndimage import label

from preprocessing.das_preprocess import (
    preprocess_das_timespace,
    compute_energy_map
)


class DASEventDetector:
    """
    Detector de eventos DAS en dominio tiempo–sensor.

    Pipeline:
        raw strain_data
            ↓
        preprocesado oficial DAS
            ↓
        mapa de energía robusto
            ↓
        umbral adaptativo
            ↓
        componentes conexas 2D
            ↓
        eventos físicos
    """

    def __init__(
        self,
        fs,
        fmin,
        fmax,
        smooth_window_sec=0.3,
        threshold=4.0,
        min_duration_sec=1.0,
        min_sensors=3,
        sigma_2d=0.8,
        clip_percentile=2
    ):
        self.fs = fs
        self.fmin = fmin
        self.fmax = fmax
        self.smooth_window_sec = smooth_window_sec
        self.threshold = threshold
        self.min_duration_sec = min_duration_sec
        self.min_sensors = min_sensors
        self.sigma_2d = sigma_2d
        self.clip_percentile = clip_percentile

    # ========================================================
    # MÉTODO PRINCIPAL
    # ========================================================

    def detect(self, X_raw, time_axis=None):
        """
        Detecta eventos en datos DAS crudos.

        Parameters
        ----------
        X_raw : ndarray (T, S)
            strain_data sin procesar
        time_axis : ndarray (T,), opcional
            vector temporal real

        Returns
        -------
        events : list[dict]
        energy_map : ndarray (T, S)
        mask : ndarray (T, S) bool
        X_proc : ndarray (T, S)
        """

        # ----------------------------------------------------
        # 1. PREPROCESADO OFICIAL
        # ----------------------------------------------------
        X_proc = preprocess_das_timespace(
            X_raw,
            fs=self.fs,
            fmin=self.fmin,
            fmax=self.fmax,
            sigma_2d=self.sigma_2d,
            clip_percentile=self.clip_percentile
        )

        # ----------------------------------------------------
        # 2. MAPA DE ENERGÍA
        # ----------------------------------------------------
        energy_map = compute_energy_map(
            X_proc,
            fs=self.fs,
            smooth_window_sec=self.smooth_window_sec
        )

        # ----------------------------------------------------
        # 3. UMBRAL
        # ----------------------------------------------------
        mask = energy_map > self.threshold

        # ----------------------------------------------------
        # 4. COMPONENTES CONEXAS 2D
        # ----------------------------------------------------
        structure = np.ones((3, 3))
        labels, n_labels = label(mask, structure=structure)

        events = []
        min_duration_samples = int(self.min_duration_sec * self.fs)

        # ----------------------------------------------------
        # 5. EXTRACCIÓN DE EVENTOS
        # ----------------------------------------------------
        for lab in range(1, n_labels + 1):
            idx = np.where(labels == lab)
            if idx[0].size == 0:
                continue

            t_min, t_max = idx[0].min(), idx[0].max()
            s_min, s_max = idx[1].min(), idx[1].max()

            duration = t_max - t_min + 1
            spatial_extent = s_max - s_min + 1

            if duration < min_duration_samples:
                continue
            if spatial_extent < self.min_sensors:
                continue

            events.append({
                "t_start": float(t_min / self.fs),
                "t_end": float(t_max / self.fs),
                "t_start_idx": int(t_min),
                "t_end_idx": int(t_max),
                "sensor_start": int(s_min),
                "sensor_end": int(s_max),
                "sensor_center": int(0.5 * (s_min + s_max)),
                "duration_sec": float(duration / self.fs),
                "n_sensors": int(spatial_extent),
                "mean_energy": float(np.mean(energy_map[idx])),
                "max_energy": float(np.max(energy_map[idx]))
            })

        return events, energy_map, mask, X_proc
