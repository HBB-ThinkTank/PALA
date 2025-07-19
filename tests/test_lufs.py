import os
import sys
import numpy as np
import subprocess

import pyloudnorm as pln

# Ensure the pala module can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pala')))

import lufs
import dynamics
import truepeak

# Set the path to the audio file
# AUDIO_FILE = "path/to/your/audio/file.wav"  # Replace with your actual file path
# AUDIO_FILE = r"C:\Users\Holge\Music\audio_pipeline_pre\test_low-amplitude.wav"
AUDIO_FILE = r"C:\Users\Holge\Downloads\Loudness_Meter_Test_Signals\LM3.wav"

# Load the audio file
waveform, sample_rate = dynamics.load_audio(AUDIO_FILE)

# LUFS mit pyloudnorm berechnen
def lufs_pyloudnorm(waveform, sample_rate):
    meter = pln.Meter(sample_rate, filter_class="DeMan")  # ITU-R BS.1770 Standard
    integrated = meter.integrated_loudness(waveform)  # LUFS Integrated
    
    # Fenster für Short-Term (3 Sek.) & Momentary (400 ms)
    short_term_blocks = int(len(waveform) / (sample_rate * 3))
    momentary_blocks = int(len(waveform) / (sample_rate * 0.4))
    
    short_term_values = [meter.integrated_loudness(waveform[i * sample_rate * 3 : (i + 1) * sample_rate * 3])
                         for i in range(short_term_blocks)]
    
    momentary_values = [meter.integrated_loudness(waveform[i * int(sample_rate * 0.4) : (i + 1) * int(sample_rate * 0.4)])
                         for i in range(momentary_blocks)]

    short_term = np.mean(short_term_values) if short_term_values else integrated
    momentary = np.mean(momentary_values) if momentary_values else integrated

    return {
        "Integrated": integrated,
        "Short-Term": short_term,
        "Momentary": momentary
    }

def get_ffmpeg_lufs(file_path: str):
    """
    Runs ffmpeg's ebur128 filter to obtain LUFS values for comparison.
    """
    command = [
        "ffmpeg", "-i", file_path, "-filter_complex", "ebur128", "-f", "null", "-"
    ]
    result = subprocess.run(command, stderr=subprocess.PIPE, text=True)

    # Extract LUFS Integrated from ffmpeg output
    lines = result.stderr.split("\n")
    integrated_lufs = None
    for i, line in enumerate(lines):
        if "Integrated loudness:" in line:
            try:
                integrated_lufs = float(lines[i + 1].split()[1])  # Die nächste Zeile enthält den Wert
            except (IndexError, ValueError):
                print("Warning: Could not parse LUFS value from ffmpeg output.")
                return None
            break

    return integrated_lufs

if __name__ == "__main__":
#    test_file = r"C:\Users\Holge\Music\audio_pipeline_pre\test_low-amplitude.wav"  # Replace with a valid path
    
    # Compute LUFS using our implementation
    pala_results = lufs.analyze_audio(AUDIO_FILE)
    rms_itu = dynamics.compute_rms_itu(waveform, sample_rate)
    
    # Compute LUFS using other tools
    lufs_pyl = lufs_pyloudnorm(waveform, sample_rate)
    ffmpeg_lufs = get_ffmpeg_lufs(AUDIO_FILE)
    
    print("\n📌 **LUFS Vergleich (ITU-R BS.1770, Pyloudnorm, Essentia, FFmpeg):**")
    print("PALA LUFS Integrated:", pala_results["lufs_integrated"])
    print("FFmpeg LUFS Integrated:", ffmpeg_lufs)
    print(f"Unser ITU-R BS.1770 RMS: {rms_itu:.2f} dBFS")
    print(f"LUFS Pyloudnorm: {lufs_pyl}")
    
    tp = truepeak.compute_true_peak(AUDIO_FILE, oversample=4)
    print(f"True Peak: {tp:.2f} dBTP")
    
    # Compare results
#    if ffmpeg_lufs is not None:
#        diff = abs(pala_results["lufs_integrated"] - ffmpeg_lufs)
#        print(f"Difference: {diff:.3f} LU")
#        assert diff < 1.0, "LUFS calculation deviates too much from ffmpeg!"
