# preprocessing/das_preprocess.py

import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import gaussian_filter, uniform_filter1d

# ============================================================
# 1. LECTURA Y DOWNSAMPLING
# ============================================================

def read_das_npz(path):
    """
    Reads DAS data stored in an NPZ file.

    Expected keys:
        - strain_data : (Nt, Nch)
        - t           : (Nt,)
        - d_total     : float
        - sampling_freq : float
    """
    data = np.load(path, allow_pickle=True)

    return {
        "strain_data": data["strain_data"],
        "t": data["t"],
        "d_total": float(data["d_total"]),
        "sampling_freq": float(data["sampling_freq"]),
    }


def downsample_das(das, target_freq):
    """
    Downsamples DAS data to a target frequency.
    """
    current_freq = das["sampling_freq"]

    if target_freq >= current_freq:
        return das

    factor = int(current_freq / target_freq)
    factor = max(factor, 1)

    das["strain_data"] = signal.decimate(
        das["strain_data"], factor, axis=0
    )
    das["t"] = das["t"][::factor]
    das["sampling_freq"] = current_freq / factor

    return das


# ============================================================
# 2. PREPROCESADO ESPACIO–TEMPORAL (OFICIAL)
# ============================================================

def remove_common_mode(X):
    """
    Elimina ruido instrumental restando la mediana espacial.
    X: (T, S)
    """
    return X - np.median(X, axis=1, keepdims=True)


def bandpass_filter_sos(X, fs, fmin, fmax, order=2):
    """
    Filtro pasabanda estable usando Second-Order Sections.
    """
    nyq = 0.5 * fs
    low = max(fmin / nyq, 1e-4)
    high = min(fmax / nyq, 0.9999)

    sos = butter(order, [low, high], btype="band", output="sos")

    # centrar para evitar transientes
    Xc = X - np.mean(X, axis=0, keepdims=True)
    return sosfiltfilt(sos, Xc, axis=0)


def apply_robust_2d_scaling(X, clip_percentile=2):
    """
    Normalización robusta global a rango [-1, 1].
    """
    p_low, p_high = np.percentile(
        X, [clip_percentile, 100 - clip_percentile]
    )

    Xc = np.clip(X, p_low, p_high)
    return 2 * (Xc - p_low) / (p_high - p_low + 1e-12) - 1


def preprocess_das_timespace(
    X,
    fs,
    fmin,
    fmax,
    sigma_2d=0.8,
    clip_percentile=2
):
    """
    Preprocesado DAS COMPLETO y UNIFICADO.

    Aplica exactamente:
    - CM removal
    - Band-pass SOS
    - Suavizado 2D
    - Scaling robusto [-1, 1]

    Parameters
    ----------
    X : ndarray (T, S)
    fs : float
    fmin, fmax : float
    sigma_2d : float
    """

    # 1. Common-mode removal
    X = remove_common_mode(X)

    # 2. Band-pass
    X = bandpass_filter_sos(X, fs, fmin, fmax)

    # 3. Suavizado espacio–temporal
    if sigma_2d > 0:
        X = gaussian_filter(X, sigma=sigma_2d)

    # 4. Scaling robusto
    X = apply_robust_2d_scaling(X, clip_percentile)

    return X


# ============================================================
# 3. UTILIDADES PARA DETECCIÓN
# ============================================================

def compute_energy_map(
    X,
    fs,
    smooth_window_sec=0.3
):
    """
    Construye mapa de energía robusto a partir de datos preprocesados.
    """
    E = np.abs(X)

    if smooth_window_sec > 0:
        win = int(smooth_window_sec * fs)
        if win > 1:
            E = uniform_filter1d(E, size=win, axis=0)

    # normalización robusta GLOBAL (para no borrar eventos largos)
    med = np.median(E)
    mad = np.median(np.abs(E - med)) + 1e-6

    return (E - med) / mad
