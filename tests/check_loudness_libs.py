import importlib
import requests

# Liste der zu prüfenden Bibliotheken
libraries = [
    "pyloudnorm",      # ITU-R BS.1770-4 Loudness Measurement
    "ffmpeg_normalize",  # FFmpeg-basierte Lautheitsnormalisierung
    "essentia",        # Audioanalyse-Framework mit Loudness-Analyse
    "loudness",        # EBU R128 Loudness-Messung (veraltet?)
    "ebur128"          # Python-Bindings für libebur128 (EBU R128-konform)
]

# Funktion zur Prüfung, ob die Bibliothek installiert ist
def is_installed(lib):
    return importlib.util.find_spec(lib) is not None

# Funktion zur Prüfung, ob die Bibliothek auf PyPI existiert
def is_on_pypi(lib):
    url = f"https://pypi.org/pypi/{lib}/json"
    response = requests.get(url)
    return response.status_code == 200

# Ergebnisse sammeln
results = {}
for lib in libraries:
    installed = is_installed(lib)
    on_pypi = is_on_pypi(lib)
    if installed:
        results[lib] = "✅ Installiert"
    elif on_pypi:
        results[lib] = "❌ Fehlend – Installierbar"
    else:
        results[lib] = "⛔ Nicht verfügbar auf PyPI"

# Ergebnisse ausgeben
print("\n📌 **Lautheits-Bibliotheken in Python – Installationsstatus:**")
for lib, status in results.items():
    print(f"{lib}: {status}")

# Falls Bibliotheken installierbar sind, pip-Befehl ausgeben
installable_libs = [lib for lib, status in results.items() if status == "❌ Fehlend – Installierbar"]
if installable_libs:
    print("\n🚀 **Falls du fehlende Bibliotheken installieren möchtest, nutze diesen Befehl:**")
    print("pip install " + " ".join(installable_libs))
