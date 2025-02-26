import sys
import os
import numpy as np

# Ensure the pala module can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pala')))

import dynamics

# Set the path to the audio file
# AUDIO_FILE = "path/to/your/audio/file.wav"  # Replace with your actual file path
# AUDIO_FILE = r"C:\Users\Holge\Music\audio_pipeline_pre\test_low-amplitude.wav"
AUDIO_FILE = r"C:\Users\Holge\Downloads\Loudness_Meter_Test_Signals\LM3.wav"

# Load the audio file
waveform, sample_rate = dynamics.load_audio(AUDIO_FILE)

# Compute values
rms_aes_value = dynamics.compute_rms_aes(waveform)
rms_itu_value = dynamics.compute_rms_itu(waveform)
peak_value = dynamics.compute_peak(waveform)
true_peak_value = dynamics.compute_true_peak(waveform)
dynamics_values = dynamics.compute_dynamics(waveform)

# Compute values directly from file
rms_aes_file = dynamics.compute_rms_aes_load(AUDIO_FILE)
rms_itu_file = dynamics.compute_rms_itu_load(AUDIO_FILE)
peak_file = dynamics.compute_peak_load(AUDIO_FILE)
true_peak_file = dynamics.compute_true_peak_load(AUDIO_FILE)
dynamics_file = dynamics.compute_dynamics_load(AUDIO_FILE)

# Print results from waveform
print("--- Results from waveform ---")
print(f"RMS AES Standard: {rms_aes_value:.2f} dBFS")
print(f"RMS ITU-R BS.1770 Standard: {rms_itu_value:.2f} dBFS")
print(f"Peak Loudest Channel: {peak_value:.2f} dBFS")
print(f"True Peak Level: {true_peak_value:.2f} dBFS")
print(f"Dynamic Range (DR): {dynamics_values['DR']:.2f} dB")
print(f"Crest Factor: {dynamics_values['Crest Factor']:.2f} dB")
print(f"Headroom: {dynamics_values['Headroom']:.2f} dB")

# Print results from direct file processing
print("\n--- Results from direct file processing ---")
print(f"RMS AES Standard: {rms_aes_file:.2f} dBFS")
print(f"RMS ITU-R BS.1770 Standard: {rms_itu_file:.2f} dBFS")
print(f"Peak Loudest Channel: {peak_file:.2f} dBFS")
print(f"True Peak Level: {true_peak_file:.2f} dBFS")
print(f"Dynamic Range (DR): {dynamics_file['DR']:.2f} dB")
print(f"Crest Factor: {dynamics_file['Crest Factor']:.2f} dB")
print(f"Headroom: {dynamics_file['Headroom']:.2f} dB")