# PALA – Python Audio Loudness Analysis

PALA is a modular Python package for measuring audio loudness according to the ITU-R BS.1770-5 standard.\
It provides LUFS (I/S/M), True Peak, RMS, DR and LRA calculations with K-weighting and gating.

---

## 🍇 Purpose

The aim of PALA is to offer a unified, open-source loudness analysis tool for:

- mastering and post-production engineers
- researchers in psychoacoustics and signal processing
- developers building automated workflows

---

## 🏢 Installation

```bash
pip install pala
```

Or from source:

```bash
git clone https://github.com/HBB-ThinkTank/PALA.git
cd PALA
pip install .
```

---

## 🧠 Features (v0.2.0)

- ✅ LUFS calculation (Integrated, Short-Term, Momentary)
- ✅ True Peak (with oversampling), Sample Peak
- ✅ RMS (AES and ITU-R variants)
- ✅ DR (Dynamic Range), Crest Factor, Headroom
- ✅ Loudness Range (LRA using 10–95 percentile)
- ✅ Chunked processing for large audio files
- ✅ Modular structure with `lufs`, `dynamics`, `utils`, `io`

---

## 📊 Development Roadmap

### Equal-Loudness Curves (Fletcher-Munson)

- 🛠️ **Integration into frequency weighting**
- 🛠️ **Application to LUFS evaluation**

### Frequency Analysis

- 🛠️ **FFT or Bark/octave band analysis**
- 🛠️ **Loudness per frequency band**

### LRA (Loudness Range)

- ✅ **LRA based on LUFS segments (10–95 percentile)**
- 🛠️ **Interchannel gating / hysteresis**

### LUFS Calculation (ITU-R BS.1770)

- ✅ **Momentary LUFS (400ms)**
- ✅ **Short-Term LUFS (3s)**
- ✅ **Integrated LUFS**
- ✅ **Gating: Absolute (-70 LUFS)**
- ✅ **Gating: Relative (-10 LU below average level)**
- ✅ **Multichannel weighting (e.g. 1.0, 1.41)**
- 🛠️ **Comparison with reference tools (e.g. pyloudnorm, ffmpeg)**
- 🛠️ **Validation using ITU-R BS.2217 material**

### Package Structure & Interfaces

- ✅ **Modular package with analysis/lufs/dynamics/utils/io**
- ✅ **Chunking support for large files**
- ✅ **CLI support for single file input**
- 🛠️ **CLI batch mode**
- 🛠️ **Plot output (e.g. LUFS-M time curve)**

### True Peak / Peak / RMS / DR

- ✅ **True Peak with oversampling**
- ✅ **Sample Peak (maximum sample value)**
- ✅ **AES RMS (unweighted)**
- ✅ **ITU-R RMS (K-weighted)**
- ✅ **Crest Factor / Headroom**
- ✅ **DR (True Peak - RMS)**

### Publishing & Distribution

- ✅ **setup.cfg / pyproject.toml**
- ✅ **GitHub with README**
- 🛠️ **Upload to PyPI (production release)**

---

## 🔮 Status

This project is in **alpha** stage. Results are close to professional tools (e.g. ffmpeg, pyloudnorm, APU plugins)\
but not yet fully validated against ITU-R BS.2217 reference material.

Expect minor deviations (especially for edge cases or extreme signals).

---

## 🛠️ Requirements

- Python 3.8+
- numpy, scipy, soundfile, soxr, numba

---

## 📄 License

MIT License – free to use, modify and redistribute.

---

## 🙌 Contributions Welcome

You’re invited to test, extend or improve the code – feel free to open issues or pull requests.

