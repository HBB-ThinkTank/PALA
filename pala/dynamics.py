import librosa
import numpy as np
import soundfile as sf
import scipy.signal as signal
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import pyloudnorm as pln
import subprocess
import os
from typing import Tuple, Optional
from utils import process_in_chunks
from functools import partial


def load_audio(filepath: str, sr: int = None, ffmpeg_path: Optional[str] = None) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file and returns the waveform and sample rate. If the format is not natively supported,
    it attempts to convert it to WAV using FFmpeg.
    
    :param filepath: Path to the audio file.
    :param sr: Target sample rate (if None, original sample rate is used).
    :param ffmpeg_path: Optional path to ffmpeg.exe if it's not in the system PATH.
    :return: Tuple (waveform, sample rate)
    """
    try:
        waveform, sample_rate = sf.read(filepath, always_2d=True)
        return waveform, sample_rate
    except Exception as e:
        print(f"soundfile could not load the file: {e}. Trying librosa as fallback...")
        try:
            waveform, sample_rate = librosa.load(filepath, sr=sr, backend="soundfile")
            return waveform, sample_rate
        except Exception as e_librosa:
            print(f"Librosa also failed: {e_librosa}")
        
        if ffmpeg_path:
            ffmpeg_cmd = ffmpeg_path
        else:
            ffmpeg_cmd = "ffmpeg"
        
        temp_wav = filepath + ".wav"
        try:
            subprocess.run([ffmpeg_cmd, "-i", filepath, "-ar", "44100", "-ac", "2", "-y", temp_wav], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            waveform, sample_rate = sf.read(temp_wav, always_2d=True)
            os.remove(temp_wav)
            return waveform, sample_rate
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Ensure that ffmpeg is installed and available in the system PATH, or specify its path explicitly via the ffmpeg_path parameter.")
        except Exception as ffmpeg_error:
            raise RuntimeError(f"FFmpeg conversion failed: {ffmpeg_error}")


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


def compute_rms_itu(waveform: np.ndarray, db: bool = True) -> float:
    """
    Computes the RMS level using the ITU-R BS.1770 standard (LUFS-like measurement) with chunk processing.
    
    :param waveform: Audio signal as a NumPy array.
    :param db: If True, returns values in dBFS, else in linear scale.
    :return: RMS value in dBFS or linear scale.
    """
    sample_rate = 44100  # Default sample rate assumption

    # Compute K-Weighting filter coefficients (ITU-R BS.1770)
    def compute_k_filter_coefficients(fs):
        f0_hp = 38.13547087613982
        Q_hp = 0.5003270373253953
        w0_hp = 2 * np.pi * f0_hp / fs
        alpha_hp = np.sin(w0_hp) / (2 * Q_hp)

        b_hp = [1 + alpha_hp, -2 * np.cos(w0_hp), 1 - alpha_hp]
        a_hp = [1 + alpha_hp, -2 * np.cos(w0_hp), 1 - alpha_hp]

        f0_shelving = 1681.9744509555319
        Q_shelving = 0.7071752369554193
        A_shelving = 10**(4.0 / 40)
        w0_shelving = 2 * np.pi * f0_shelving / fs
        alpha_shelving = np.sin(w0_shelving) / (2 * Q_shelving)

        b_shelving = [
            A_shelving * ((A_shelving + 1) + (A_shelving - 1) * np.cos(w0_shelving) + 2 * np.sqrt(A_shelving) * alpha_shelving),
            -2 * A_shelving * ((A_shelving - 1) + (A_shelving + 1) * np.cos(w0_shelving)),
            A_shelving * ((A_shelving + 1) + (A_shelving - 1) * np.cos(w0_shelving) - 2 * np.sqrt(A_shelving) * alpha_shelving)
        ]
        a_shelving = [
            (A_shelving + 1) - (A_shelving - 1) * np.cos(w0_shelving) + 2 * np.sqrt(A_shelving) * alpha_shelving,
            2 * ((A_shelving - 1) - (A_shelving + 1) * np.cos(w0_shelving)),
            (A_shelving + 1) - (A_shelving - 1) * np.cos(w0_shelving) - 2 * np.sqrt(A_shelving) * alpha_shelving
        ]
        return b_hp, a_hp, b_shelving, a_shelving

    b_hp, a_hp, b_shelving, a_shelving = compute_k_filter_coefficients(sample_rate)
    
    # Apply K-weighting filters
    waveform = signal.lfilter(b_hp, a_hp, waveform, axis=0)
    waveform = signal.lfilter(b_shelving, a_shelving, waveform, axis=0)
    
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


def compute_true_peak(waveform: np.ndarray, oversample_factor: int = 4, db: bool = True) -> float:
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


def compute_dynamics(waveform: np.ndarray, db: bool = True, norm: str = 'aes') -> dict:
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
    rms = compute_rms_aes(waveform, db=db) if norm == 'aes' else compute_rms_itu(waveform, db=db)

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



def compute_rms_aes_load(filepath: str, sr: int = None, db: bool = True, ffmpeg_path: Optional[str] = None) -> float:
    waveform, _ = load_audio(filepath, sr=sr, ffmpeg_path=ffmpeg_path)
    return compute_rms_aes(waveform, db=db)


def compute_rms_itu_load(filepath: str, sr: int = None, db: bool = True, ffmpeg_path: Optional[str] = None) -> float:
    waveform, _ = load_audio(filepath, sr=sr, ffmpeg_path=ffmpeg_path)
    return compute_rms_itu(waveform, db=db)


def compute_peak_load(filepath: str, sr: int = None, db: bool = True, ffmpeg_path: Optional[str] = None) -> float:
    waveform, _ = load_audio(filepath, sr=sr, ffmpeg_path=ffmpeg_path)
    return compute_peak(waveform, db=db)


def compute_true_peak_load(filepath: str, sr: int = None, oversample_factor: int = 4, db: bool = True, ffmpeg_path: Optional[str] = None) -> float:
    waveform, _ = load_audio(filepath, sr=sr, ffmpeg_path=ffmpeg_path)
    return compute_true_peak(waveform, oversample_factor=oversample_factor, db=db)


def compute_dynamics_load(filepath: str, sr: int = None, db: bool = True, ffmpeg_path: Optional[str] = None) -> dict:
    waveform, _ = load_audio(filepath, sr=sr, ffmpeg_path=ffmpeg_path)
    return compute_dynamics(waveform, db=db)


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
    dynamics = compute_dynamics(rms, peak, true_peak)
    
    print(f"RMS Level: {rms:.2f} dBFS")
    print(f"Peak Level: {peak:.2f} dBFS")
    print(f"True Peak Level: {true_peak:.2f} dBFS")
    print(f"Dynamic Range (DR): {dynamics['DR']:.2f} dB")
    print(f"Crest Factor: {dynamics['Crest Factor']:.2f} dB")
    print(f"Headroom: {dynamics['Headroom']:.2f} dB")
