# import librosa
import numpy as np
# import soundfile as sf
import scipy.signal as signal
from scipy.signal import lfilter, resample_poly
# import subprocess
# import os
from typing import Tuple, Optional
from utils import process_in_chunks, apply_k_weighting_filter, load_audio
from functools import partial


def compute_rms_aes(waveform: np.ndarray, db: bool = True) -> float:
    """
    Computes the RMS level using the AES standard.

    :param waveform: Audio signal as a NumPy array.
    :param db: If True, returns values in dBFS, else in linear scale.
    :return: RMS value in dBFS or linear scale.
    """
    if waveform.ndim == 1:
        rms = np.sqrt(np.mean(np.square(waveform)))  # Mono-Fall
    else:
        rms = np.mean([np.sqrt(np.mean(np.square(waveform[:, 0]))), np.sqrt(np.mean(np.square(waveform[:, 1])))])  # AES Standard

    return 20 * np.log10(rms) if db and rms > 0 else rms


def compute_rms_itu(waveform: np.ndarray, sample_rate: int, db: bool = True) -> float:
    """
    Computes the RMS level using the ITU-R BS.1770 standard (LUFS-like measurement) with chunk processing.
    
    :param waveform: Audio signal as a NumPy array.
    :param db: If True, returns values in dBFS, else in linear scale.
    :return: RMS value in dBFS or linear scale.
    """

    waveform = apply_k_weighting_filter(waveform, sample_rate)

    # Process in chunks with 50% Overlapping
    chunk_size = int(0.4 * sample_rate)  # 400 ms window
    hop_size = chunk_size // 2  # 50% Overlapping
    num_chunks = (len(waveform) - chunk_size) // hop_size + 1
    
    rms_values = [
        np.sqrt(np.mean(np.square(waveform[i * hop_size : i * hop_size + chunk_size])))
        for i in range(num_chunks)
    ]
    
    # Convert linear values to dBFS
    rms_db_values = [20 * np.log10(r) if r > 0 else -np.inf for r in rms_values]
    
    # Energy-based averaging as per ITU-R BS.1770
    energy_sum = np.sum(np.power(10, np.array(rms_db_values) / 10))
    rms = 10 * np.log10(energy_sum / len(rms_db_values))
    
    return rms

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


def compute_true_peak_old(waveform: np.ndarray, oversample_factor: int = 4, db: bool = True) -> float:
    """
    Computes the True Peak Level using oversampling.
    
    :param waveform: Audio signal as a NumPy array.
    :param oversample_factor: Oversampling factor (default: 4x).
    :param db: If True, returns value in dBFS, else returns absolute peak.
    :return: True Peak value in dBFS or linear scale.
    """
    from scipy.signal import resample
    oversampled_waveform = resample(waveform, len(waveform) * oversample_factor)
    true_peak = np.max(np.abs(oversampled_waveform))
    return 20 * np.log10(true_peak) if db and true_peak > 0 else true_peak

def compute_true_peak(waveform: np.ndarray, oversample_factor: int = 4, db: bool = True) -> float:
    """
    Computes the True Peak level of an audio file in dBTP using oversampling.

    Parameters:
        waveform : str
            Audio signal as a NumPy array.
        oversample_factor    : int
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
        upsampled = resample_poly(waveform[:, ch], oversample_factor, 1)
        peak = np.max(np.abs(upsampled))
        true_peaks.append(peak)

    max_peak = max(true_peaks)
    return 20 * np.log10(max_peak) if db and max_peak > 0 else max_peak


def compute_dynamics(waveform: np.ndarray, sample_rate: int = 44100, db: bool = True, norm: str = 'aes') -> dict:
    """
    Computes dynamic range metrics using either AES RMS or ITU-R BS.1770 RMS.

    :param waveform: Audio signal as a NumPy array.
    :param db: If True, returns values in dB, else in linear scale.
    :param norm: Defines the RMS normalization standard ('aes' or 'itu').
    :return: Dictionary containing DR, Crest Factor, Headroom.
    """
    # Validate normalization type
    norm = norm.lower()
    if norm not in {'aes', 'itu'}:
        raise ValueError("Invalid norm type. Use 'aes' for unweighted RMS or 'itu' for ITU-R BS.1770 K-weighted RMS.")

    # Select RMS computation based on norm
    rms = compute_rms_aes(waveform, db=db) if norm == 'aes' else compute_rms_itu(waveform, sample_rate, db=db)

    # Compute peak values
    peak = compute_peak(waveform, db=db)
    true_peak = compute_true_peak(waveform, db=db)

    # Compute dynamic metrics
    if db:
        dr = true_peak - rms
        crest_factor = peak - rms
        headroom = 0 - true_peak
    else:
        dr = true_peak / rms if rms > 0 else 0
        crest_factor = peak / rms if rms > 0 else 0
        headroom = 1 - true_peak

    return {
        "DR": dr,
        "Crest Factor": crest_factor,
        "Headroom": headroom
    }

def compute_lra(waveform: np.ndarray, sample_rate: int) -> float:
    """
    Computes the Loudness Range (LRA) from short-term LUFS values (3s windows).

    Parameters:
        waveform : np.ndarray
            Mono or summed K-weighted signal
        sample_rate : int
            Sample rate in Hz

    Returns:
        float : Loudness Range (LRA) in LU
    """
    # Apply K-weighting
    waveform = apply_k_weighting_filter(waveform, sample_rate)

    # Reduce to mono if multichannel
    if waveform.ndim == 2:
        waveform = np.mean(waveform, axis=1)

    window_ms = 3000
    chunk_size = int(sample_rate * window_ms / 1000)
    hop_size = int(chunk_size * 0.25)  # 75% overlap

    lufs_values = []
    for start in range(0, len(waveform) - chunk_size + 1, hop_size):
        chunk = waveform[start:start + chunk_size]
        ms = np.mean(chunk ** 2)
        if ms > 1e-7:  # Apply absolute gate (-70 LUFS)
            lufs = -0.691 + 10 * np.log10(ms)
            lufs_values.append(lufs)

    if len(lufs_values) < 2:
        return 0.0  # Not enough data to compute range

    lufs_array = np.array(lufs_values)
    lower = np.percentile(lufs_array, 10)
    upper = np.percentile(lufs_array, 95)

    return upper - lower


def compute_rms_aes_load(filepath: str, sr: int = None, db: bool = True, ffmpeg_path: Optional[str] = None) -> float:
    waveform, _ = load_audio(filepath, sr=sr, ffmpeg_path=ffmpeg_path)
    return compute_rms_aes(waveform, db=db)


def compute_rms_itu_load(filepath: str, sr: int = None, db: bool = True, ffmpeg_path: Optional[str] = None) -> float:
    waveform, sample_rate = load_audio(filepath, sr=sr, ffmpeg_path=ffmpeg_path)
    return compute_rms_itu(waveform, sample_rate, db=db)


def compute_peak_load(filepath: str, sr: int = None, db: bool = True, ffmpeg_path: Optional[str] = None) -> float:
    waveform, _ = load_audio(filepath, sr=sr, ffmpeg_path=ffmpeg_path)
    return compute_peak(waveform, db=db)


def compute_true_peak_load(filepath: str, sr: int = None, oversample_factor: int = 4, db: bool = True, ffmpeg_path: Optional[str] = None) -> float:
    waveform, _ = load_audio(filepath, sr=sr, ffmpeg_path=ffmpeg_path)
    return compute_true_peak(waveform, oversample_factor=oversample_factor, db=db)


def compute_dynamics_load(filepath: str, sr: int = None, db: bool = True, ffmpeg_path: Optional[str] = None) -> dict:
    waveform, sample_rate = load_audio(filepath, sr=sr, ffmpeg_path=ffmpeg_path)
    return compute_dynamics(waveform, sample_rate, db=db)
    
def compute_lra_load(filepath: str, sr: int = None, ffmpeg_path: Optional[str] = None) -> float:
    waveform, sample_rate = load_audio(filepath, sr=sr, ffmpeg_path=ffmpeg_path)
    return compute_lra(waveform, sample_rate)


# Define alternative short aliases only for external usage
# Standardfunktionen
rmsaes = compute_rms_aes
rmsitu = compute_rms_itu
peak = compute_peak
true_peak = compute_true_peak
dynamics = compute_dynamics
lrmsaes = compute_rms_aes_load
lrmsitu = compute_rms_itu_load
lpeak = compute_peak_load
ltrue_peak = compute_true_peak_load
ldynamics = compute_dynamics_load
lra = compute_lra

# Funktionen mit vordefinierten Parametern (Aliases mit fixem `norm`)
dynaes = partial(compute_dynamics, norm='aes')
dynitu = partial(compute_dynamics, norm='itu')

ldynaes = partial(compute_dynamics_load, norm='aes')
ldynitu = partial(compute_dynamics_load, norm='itu')


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python dynamics.py <audio_file>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    waveform, sr = load_audio(audio_path)
    
    rms = compute_rms(waveform)
    peak = compute_peak(waveform)
    true_peak = compute_true_peak(waveform)
    dynamics_result = compute_dynamics(waveform, sr)
    lra = compute_lra(waveform)
    
    print(f"RMS Level: {rms:.2f} dBFS")
    print(f"Peak Level: {peak:.2f} dBFS")
    print(f"True Peak Level: {true_peak:.2f} dBFS")
    print(f"Dynamic Range (DR): {dynamics_result['DR']:.2f} dB")
    print(f"Crest Factor: {dynamics['Crest Factor']:.2f} dB")
    print(f"Headroom: {dynamics['Headroom']:.2f} dB")
    print(f"Loudness Range: {lra:.2f} LU")

