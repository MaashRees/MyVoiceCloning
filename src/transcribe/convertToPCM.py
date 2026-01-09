import os
import soundfile as sf
from pyprojroot import here

ROOT_DIR = here()

def convert_to_pcm(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    print(f"🔄 Conversion de {len(files)} fichiers en cours...")

    for filename in files:
        path = os.path.join(folder_path, filename)
        data, fs = sf.read(path)
        sf.write(path, data, fs, subtype='PCM_16')
        print(f"✅ Corrigé : {filename}")

if __name__ == "__main__":
    mon_dossier = os.path.join(ROOT_DIR, "data", "enregistrements", "20260109_001656")
    convert_to_pcm(mon_dossier)