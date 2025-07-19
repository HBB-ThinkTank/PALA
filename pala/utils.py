import numpy as np
from typing import Tuple, Optional, Callable, Any, Generator
import soundfile as sf
import scipy.signal


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


def process_in_chunks(
    waveform: np.ndarray,
    sample_rate: int,
    chunk_size_ms: int,
    process_func: Callable[[np.ndarray, int], Any]
) -> Generator[Any, None, None]:
    """
    Processes an audio waveform in chunks and applies a given function to each chunk.
    
    :param waveform: The input audio signal as a NumPy array.
    :param sample_rate: The sample rate of the audio signal.
    :param chunk_size_ms: The chunk size in milliseconds.
    :param process_func: A function that processes each chunk (expects waveform chunk and sample_rate).
    :return: A generator that yields results from process_func for each chunk.
    """
    chunk_size = int((chunk_size_ms / 1000) * sample_rate)  # Convert ms to samples
    num_samples = waveform.shape[0]
    
    for start in range(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)
        chunk = waveform[start:end]
        yield process_func(chunk, sample_rate)


# Example Usage: Compute RMS per chunk

def compute_rms_chunk(chunk: np.ndarray, sample_rate: int) -> float:
    """Computes the RMS of a given audio chunk."""
    rms = np.sqrt(np.mean(np.square(chunk)))
    return 20 * np.log10(rms) if rms > 0 else -np.inf


# Example on how to use process_in_chunks()

# Compute K-Weighting filter coefficients (ITU-R BS.1770)
def apply_k_weighting_filter(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Applies the ITU-R BS.1770 K-Weighting filter (High-Pass + Shelving) to an audio signal.

    :param waveform: Input audio signal as a NumPy array.
    :param sample_rate: Sampling rate of the signal.
    :return: Filtered waveform after applying K-Weighting.
    """
    # DEBUG print(f"Sample Rate vor Filterung: {sample_rate}")
    # ITU-R BS.1770 Standardwerte
    G = 4  # Verstärkung des Shelving-Filters in dB
    high_pass_Q = 0.5
    high_shelf_Q = 1/np.sqrt(2)
    high_pass_fc = 38  # Grenzfrequenz Hochpass
    high_shelf_fc = 1500  # Grenzfrequenz Shelving

    # Berechnung der Verstärkung in linearer Skala
    A = 10**(G/40.0)

    # Berechnung der normalisierten Kreisfrequenz
    high_pass_w0 = 2.0 * np.pi * (high_pass_fc / sample_rate)
    high_shelf_w0 = 2.0 * np.pi * (high_shelf_fc / sample_rate)

    # Berechnung der Filtersteilheit (Bandbreite)
    high_pass_alpha = np.sin(high_pass_w0) / (2.0 * high_pass_Q)
    high_shelf_alpha = np.sin(high_shelf_w0) / (2.0 * high_shelf_Q)

    # Shelving-Filter-Koeffizienten
    high_shelf_b0 = A * ((A + 1) + (A - 1) * np.cos(high_shelf_w0) + 2 * np.sqrt(A) * high_shelf_alpha)
    high_shelf_b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(high_shelf_w0))
    high_shelf_b2 = A * ((A + 1) + (A - 1) * np.cos(high_shelf_w0) - 2 * np.sqrt(A) * high_shelf_alpha)
    high_shelf_a0 = (A + 1) - (A - 1) * np.cos(high_shelf_w0) + 2 * np.sqrt(A) * high_shelf_alpha
    high_shelf_a1 = 2 * ((A - 1) - (A + 1) * np.cos(high_shelf_w0))
    high_shelf_a2 = (A + 1) - (A - 1) * np.cos(high_shelf_w0) - 2 * np.sqrt(A) * high_shelf_alpha

    # Normalisierung Shelving-Filter
    highshelf_b = [high_shelf_b0 / high_shelf_a0, high_shelf_b1 / high_shelf_a0, high_shelf_b2 / high_shelf_a0]
    highshelf_a = [1.0, high_shelf_a1 / high_shelf_a0, high_shelf_a2 / high_shelf_a0]

    # DEBUG print("\nSollwerte bei 48000 Hz für Shelving-Filter:")
    # DEBUG print("b0 1.53512485958697\nb1 −2.69169618940638\nb2 1.19839281085285")
    # DEBUG print("a0: 1\na1: −1.69065929318241\na2 0.73248077421585")
    # DEBUG print("\nErrechnet für Shelving-Filter:")
    # DEBUG print(f"b0 {high_shelf_b0 / high_shelf_a0}\nb1 {high_shelf_b1 / high_shelf_a0}\nb2 {high_shelf_b2 / high_shelf_a0}")
    # DEBUG print(f"a0 1\nb1 {high_shelf_a1 / high_shelf_a0}\na2 {high_shelf_a2 / high_shelf_a0}")

    # Hochpassfilter-Koeffizienten
    high_pass_b0 = (1 + np.cos(high_pass_w0)) / 2
    high_pass_b1 = -(1 + np.cos(high_pass_w0))
    high_pass_b2 = (1 + np.cos(high_pass_w0)) / 2
    high_pass_a0 = 1 + high_pass_alpha
    high_pass_a1 = -2 * np.cos(high_pass_w0)
    high_pass_a2 = 1 - high_pass_alpha
    
    # DEBUG print("\nSollwerte bei 48000 Hz für Hochpassfilter:")
    # DEBUG print("b0 1\nb1 −2\nb2 1")
    # DEBUG print("a0: 1\na1: −1.99004745483398\na2 0.99007225036621")

    # Normalisierung Hochpassfilter
    highpass_b = [high_pass_b0 / high_pass_a0, high_pass_b1 / high_pass_a0, high_pass_b2 / high_pass_a0]
    # highpass_b = [1.535, -2.69, 1.2]
    highpass_a = [1.0, high_pass_a1 / high_pass_a0, high_pass_a2 / high_pass_a0]
    # highpass_a = [1, -1.69, 0.732]
    
    # DEBUG print("\nSample Rate nach Hochfilter vor Shelving-Filter: ", sample_rate)
    # DEBUG print("\nErrechnet für Hochpassfilter:")
    # DEBUG print(f"b0 {high_pass_b0 / high_pass_a0}\nb1 {high_pass_b1 / high_pass_a0}\nb2 {high_pass_b2 / high_pass_a0}")
    # DEBUG print(f"a0 1\nb1 {high_pass_a1 / high_pass_a0}\na2 {high_pass_a2 / high_pass_a0}")

    # Anwendung des Shelving-Filters
    waveform_hp = scipy.signal.lfilter(highshelf_b, highshelf_a, waveform)

    # Anwendung des Hochpassfilters
    waveform_filtered = scipy.signal.lfilter(highpass_b, highpass_a, waveform_hp)

    return waveform_filtered.astype(np.float32)

if __name__ == "__main__":
    import soundfile as sf
    
    # Load a test file
    waveform, sample_rate = sf.read("path/to/audio.wav")
    
    # Process in 400ms chunks and compute RMS per chunk
    rms_values = list(process_in_chunks(waveform, sample_rate, 400, compute_rms_chunk))
    
    # Print results
    print("RMS per chunk:", rms_values)
