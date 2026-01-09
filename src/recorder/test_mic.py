from time import sleep

import sounddevice as sd
import numpy as np


def tester_qualite(device_id, label):
    fs = 44100
    try:
        print(f"\n--- Test de : {label} (Port {device_id}) ---")
        sd.check_input_settings(device=device_id, samplerate=fs)
        print("✅ Ce port supporte la Haute Fidélité (44.1kHz).")

        print("🎤 Enregistrement test de 10 secondes... Parlez dans deux secondes !")
        sleep(5)
        audio = sd.rec(int(10 * fs), samplerate=fs, channels=1, device=device_id)
        sd.wait()

        volume = np.sqrt(np.mean(audio ** 2))
        if volume < 0.001:
            print("⚠️ Attention : Le son semble très faible ou muet.")
        else:
            print(f"📊 Volume capté : {volume:.4f} (OK)")

    except Exception as e:
        print(f"❌ Port {device_id} inutilisable pour la haute qualité : {e}")


candidats = [
    (1, "Casque Sennheiser"),
    (37, "Micro Ordi (Realtek)"),
    (2, "DJI Mic (Mode dégradé Hands-Free)")
]

for p_id, p_name in candidats:
    tester_qualite(p_id, p_name)