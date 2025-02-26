import numpy as np
from typing import Callable, Any, Generator


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
if __name__ == "__main__":
    import soundfile as sf
    
    # Load a test file
    waveform, sample_rate = sf.read("path/to/audio.wav")
    
    # Process in 400ms chunks and compute RMS per chunk
    rms_values = list(process_in_chunks(waveform, sample_rate, 400, compute_rms_chunk))
    
    # Print results
    print("RMS per chunk:", rms_values)
