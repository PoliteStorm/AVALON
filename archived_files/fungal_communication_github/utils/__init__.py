# Utility helpers for research-grade testing

import os
import numpy as np
import pandas as pd

__all__ = [
    'load_voltage_trace',
    'load_spike_times',
    'generate_pink_noise',
    'apply_environmental_noise'
]

_DATA_DIR = os.environ.get('FUNGAL_DATA_DIR', 'data')


def _full_path(filename: str) -> str:
    """Return absolute path to a data file inside the data directory."""
    return os.path.join(_DATA_DIR, filename)


def load_voltage_trace(filename: str) -> np.ndarray:
    """Load voltage trace CSV (single column) as numpy array.
    Raises FileNotFoundError if file missing.
    """
    path = _full_path(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, header=None).values.flatten()


def load_spike_times(filename: str) -> np.ndarray:
    """Load ground-truth spike times (seconds) CSV as numpy array."""
    path = _full_path(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, header=None).values.flatten()

# ---------------- Environmental noise helpers ---------------- #

def _voss(num_rows: int, num_cols: int) -> np.ndarray:
    """Generate pink noise using Voss-McCartney algorithm."""
    array = np.random.randn(num_rows, num_cols)
    array = np.cumsum(array, axis=0)
    denom = np.power(2, np.arange(num_rows))[:, None]
    return (array / denom).sum(axis=0)


def generate_pink_noise(n_samples: int, amplitude: float = 1.0) -> np.ndarray:
    """Return pink noise of given length and amplitude."""
    rows = int(np.ceil(np.log2(n_samples)))
    noise = _voss(rows, n_samples)
    noise = noise / np.max(np.abs(noise))  # normalise
    return noise * amplitude


def apply_environmental_noise(signal: np.ndarray, pink_amp: float = 1e-5,
                               drift_amp: float = 1e-4) -> np.ndarray:
    """Add pink noise and slow baseline drift to a voltage trace."""
    n = len(signal)
    pink = generate_pink_noise(n, pink_amp)
    t = np.linspace(0, 1, n)
    drift = drift_amp * np.sin(2 * np.pi * 0.01 * t)  # very low-freq drift
    return signal + pink + drift 