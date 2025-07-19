import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

def compute_true_peak(waveform: np.ndarray, oversample: int = 4, db: bool = True) -> float:
    """
    Computes the True Peak level of an audio file in dBTP using oversampling.

    Parameters:
        waveform : str
            Audio signal as a NumPy array.
        oversample : int
            Oversampling factor (commonly 4 or 8).
        db : bool
            If True, returns values in dBTP, else in linear scale.

    Returns:
        float : True Peak value in dBTP (decibels relative to full scale) or linear scale.
    """

    # Ensure 2D shape: (samples, channels)
    if waveform.ndim == 1:
        waveform = waveform[:, np.newaxis]

    true_peaks = []
    for ch in range(waveform.shape[1]):
        # Resample with polyphase filtering
        upsampled = resample_poly(waveform[:, ch], oversample, 1)
        peak = np.max(np.abs(upsampled))
        true_peaks.append(peak)

    max_peak = max(true_peaks)
    return 20 * np.log10(max_peak) if db and max_peak > 0 else max_peak


def compute_peak(waveform: np.ndarray, db: bool = True) -> float:
    """
    Computes the Peak Level using the loudest channel.
    
    :param waveform: Audio signal as a NumPy array.
    :param db: If True, returns values in dBFS, else in linear scale.
    :return: Peak value in dBFS or linear scale.
    """
    if waveform.ndim == 1:
        peak = np.max(np.abs(waveform))  # Mono-Fall
    else:
        peak = np.max([np.max(np.abs(waveform[:, 0])), np.max(np.abs(waveform[:, 1]))])  # Lautester Kanal

    return 20 * np.log10(peak) if db and peak > 0 else peak

