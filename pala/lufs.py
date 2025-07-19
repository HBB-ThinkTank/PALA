import numpy as np
import scipy.signal as signal
import soundfile as sf
from utils import apply_k_weighting_filter, process_in_chunks

def compute_mean_square(chunk: np.ndarray) -> float:
    """
    Compute the mean square (power) of a signal chunk.
    """
    return np.mean(chunk ** 2)

def compute_gated_loudness_ms(data: np.ndarray, rate: int, chunk_size_ms: int = 400) -> float:
    """
    Compute gated loudness according to ITU-R BS.1770 (Mono, K-weighted, single channel).

    Parameters:
        data: np.ndarray – mono audio signal (after K-weighting)
        rate: int – sample rate
        chunk_size_ms: int – window size in ms (default 400ms as per ITU)

    Returns:
        Loudness in LUFS or -inf if all chunks are gated out
    """
    # Step 1: Zerlege in überlappende 400ms-Chunks mit 75% Overlap
    hop_size = int(rate * chunk_size_ms / 1000 * 0.25)
    chunk_size = int(rate * chunk_size_ms / 1000)
    
    chunks = []
    for start in range(0, len(data) - chunk_size + 1, hop_size):
        chunk = data[start:start + chunk_size]
        chunks.append(chunk)

    # Step 2: Berechne Mean Square (MS) pro Chunk
    ms_values = [compute_mean_square(chunk) for chunk in chunks]

    # Step 3: Absolute Gating (-70 LUFS ≈ 10^(-7))
    gated_ms = [ms for ms in ms_values if ms > 1e-7]
    if not gated_ms:
        return -np.inf  # Alles rausgefiltert

    # Step 4: Relative Gating (10 dB unter Mittelwert)
    gated_array = np.array(gated_ms)
    mean_ms = np.mean(gated_array)
    rel_threshold = mean_ms / 10  # -10 dB = 1/10 Leistung

    gated_final = gated_array[gated_array > rel_threshold]
    if not len(gated_final):
        return -np.inf

    # Step 5: Finaler LUFS-Wert aus Mittelwert der gültigen MS-Blöcke
    gated_mean = np.mean(gated_final)
    loudness = -0.691 + 10 * np.log10(gated_mean)

    return loudness

# Compute RMS for each chunk
def compute_rms_chunk(chunk: np.ndarray, sample_rate: int) -> float:
    """
    Computes the RMS of a given audio chunk.
    """
    rms = np.sqrt(np.mean(np.square(chunk)))
    return 20 * np.log10(rms) if rms > 0 else -np.inf

# Gated RMS Calculation
def compute_gated_rms_old(data: np.ndarray, rate: int, chunk_size_ms: int) -> float:
    """
    Computes the gated RMS according to ITU-R BS.1770.
    """
    rms_values = list(process_in_chunks(data, rate, chunk_size_ms, compute_rms_chunk))
    
    # Absolute gate (-70 LUFS)
    gated_rms_values = [rms for rms in rms_values if rms > -70]
    
    if not gated_rms_values:
        return -np.inf  # If all values are gated out, return silence indicator
    
    # Compute average loudness and apply relative gate (-10 LU below mean)
    mean_loudness = np.mean(gated_rms_values)
    gated_rms_values = [rms for rms in gated_rms_values if rms > mean_loudness - 10]
    
    return np.mean(gated_rms_values) if gated_rms_values else -np.inf

def compute_gated_rms(data: np.ndarray, rate: int, chunk_size_ms: int) -> float:
    """
    Computes LUFS for short-term or momentary loudness according to ITU-R BS.1770-5.
    Gating and mean square power are applied correctly (no dB-based gating).
    
    Parameters:
        data: np.ndarray – mono signal (after K-weighting)
        rate: int – sample rate in Hz
        chunk_size_ms: int – analysis window in milliseconds (e.g. 3000 or 400)
    
    Returns:
        LUFS value or -inf if all chunks are gated out
    """
    # Step 1: define chunk and hop size (75% overlap)
    chunk_size = int(rate * chunk_size_ms / 1000)
    hop_size = int(chunk_size * 0.25)
    
    # Step 2: slice into overlapping chunks
    chunks = []
    for start in range(0, len(data) - chunk_size + 1, hop_size):
        chunks.append(data[start:start + chunk_size])
    
    if not chunks:
        return -np.inf

    # Step 3: compute mean square of each chunk
    ms_values = [np.mean(chunk ** 2) for chunk in chunks]

    # Step 4: absolute gate at 10^(-7) = -70 LUFS
    ms_values_abs_gated = [ms for ms in ms_values if ms > 1e-7]
    if not ms_values_abs_gated:
        return -np.inf

    # Step 5: relative gate at -10 LU (1/10 of average energy)
    avg_ms = np.mean(ms_values_abs_gated)
    rel_threshold = avg_ms / 10
    ms_values_final = [ms for ms in ms_values_abs_gated if ms > rel_threshold]

    if not ms_values_final:
        return -np.inf

    # Step 6: calculate final LUFS value
    gated_avg = np.mean(ms_values_final)
    lufs = -0.691 + 10 * np.log10(gated_avg)

    return lufs


def compute_lufs_integrated_multichannel(data: np.ndarray, rate: int, channel_weights: list[float]) -> float:
    """
    Computes Integrated LUFS value across all channels according to ITU-R BS.1770-5.
    
    Parameters:
        data : np.ndarray
            K-weighted multichannel audio signal [samples x channels].
        rate : int
            Sample rate in Hz.
        channel_weights : list of float
            Channel gain factors according to ITU (e.g. [1.0, 1.0] for stereo).
    
    Returns:
        float : Integrated LUFS or -inf if gated out.
    """
    chunk_size_ms = 400
    chunk_size = int(rate * chunk_size_ms / 1000)
    hop_size = int(chunk_size * 0.25)

    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)

    n_channels = data.shape[1]
    assert len(channel_weights) == n_channels, "Mismatch in channel count and weight list!"

    # Step 1: Mean square per chunk per channel
    z_i_chunks = [[] for _ in range(n_channels)]

    for start in range(0, len(data) - chunk_size + 1, hop_size):
        chunk = data[start:start + chunk_size]
        for ch in range(n_channels):
            ms = compute_mean_square(chunk[:, ch])
            z_i_chunks[ch].append(ms)

    # Step 2: Sum across channels with weights → z_total_chunk[j]
    z_total_chunks = []
    for i in range(len(z_i_chunks[0])):
        z_sum = 0
        for ch in range(n_channels):
            z_sum += channel_weights[ch] * z_i_chunks[ch][i]
        z_total_chunks.append(z_sum)

    # Step 3: Absolute gating (-70 LUFS = 1e-7)
    gated_chunks = [z for z in z_total_chunks if z > 1e-7]
    if not gated_chunks:
        return -np.inf

    # Step 4: Relative gating (-10 LU = 1/10 linear power)
    mean_z = np.mean(gated_chunks)
    threshold = mean_z / 10
    final_chunks = [z for z in gated_chunks if z > threshold]
    if not final_chunks:
        return -np.inf

    # Step 5: Final LUFS
    z_avg = np.mean(final_chunks)
    lufs = -0.691 + 10 * np.log10(z_avg)

    return lufs


# Compute LUFS Integrated
def compute_lufs_integrated(data: np.ndarray, rate: int) -> float:
    """
    Computes the Integrated LUFS value based on ITU-R BS.1770.
    """
    rms_integrated = compute_gated_rms(data, rate, chunk_size_ms=3000)  # Use 3s blocks
    
    if rms_integrated == -np.inf:
        return -np.inf  # Return silence indicator if all blocks are gated out
    
    return rms_integrated - 0.691  # Apply ITU-R correction factor

# Main analysis module
def analyze_audio(file_path: str) -> dict:
    """
    Analyzes an audio file and applies K-weighting filter.
    """
    data, rate = sf.read(file_path)
    print(f"Shape: {data.shape}, dtype: {data.dtype}, max: {np.max(data):.3f}")
    data = apply_k_weighting_filter(data, rate)

    # Stereo = [L, R] → beide Kanäle mit Gewicht 1.0
    weights = [1.0, 1.0] if data.ndim == 2 and data.shape[1] == 2 else [1.0]

    lufs_integrated = compute_lufs_integrated_multichannel(data, rate, weights)
#   lufs_integrated = compute_lufs_integrated(data, rate)
    lufs_short_term = compute_gated_rms(data, rate, 3000)  # Short-Term LUFS (3s window)
    lufs_momentary = compute_gated_rms(data, rate, 400)   # Momentary LUFS (400ms window)
    
    return {
        "filename": file_path,
        "lufs_integrated": lufs_integrated,
        "lufs_short_term": lufs_short_term,
        "lufs_momentary": lufs_momentary
    }

# Main execution block
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pala.py <audio_file>")
    else:
        file_path = sys.argv[1]
        result = analyze_audio(file_path)
        print(result)
