import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import os
import time
from datetime import datetime
import threading
from pyprojroot import here


ROOT_DIR = here()

def list_microphones():
    """
        Affiche la liste des périphériques audio d'entrée disponibles sur le système.

        Cette fonction interroge le système via la bibliothèque 'sounddevice' pour
        récupérer tous les périphériques connectés. Elle filtre les résultats pour
        n'afficher que ceux possédant au moins un canal d'entrée (microphones).

        Returns:
            list: Une liste de dictionnaires contenant les propriétés de chaque
                  périphérique (nom, nombre de canaux, fréquence d'échantillonnage par défaut, etc.).
    """
    print("\n--- Micros disponibles ---")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"[{i}] {dev['name']}")
    return devices


def record_audio():
    """
        Gère l'intégralité du processus d'enregistrement, de sauvegarde et de segmentation audio.

        Déroulement technique de la fonction :
        1.  **Initialisation** : Demande le nom de l'utilisateur et crée une structure de dossiers
            dynamique dans 'data/enregistrements/' à partir de la racine du projet.
        2.  **Configuration** : Permet de choisir le micro et le mode de capture (temporel ou manuel).
        3.  **Capture de flux (Threading)** :
            - Utilise 'sd.InputStream' pour lire les données audio par blocs (chunks) d'une seconde.
            - Lance un thread parallèle pour surveiller l'appui sur la touche 'Entrée' afin de
              ne pas bloquer la lecture du flux audio.
        4.  **Traitement de données** : Une fois l'enregistrement stoppé, concatène tous les
            blocs NumPy en un seul tableau de données.
        5.  **Segmentation** : Découpe l'audio complet en segments de 5 minutes (300 secondes)
            pour éviter d'avoir des fichiers .wav trop volumineux et faciliter l'analyse ultérieure.
        6.  **Export** : Sauvegarde chaque segment au format WAV avec un nommage horodaté.

        Note :
            La fréquence d'échantillonnage est fixée à 44100 Hz pour une qualité CD.
    """
    nom = input("\nEntrez le nom de la personne à enregistrer : ").strip().replace(" ", "_")

    devices = list_microphones()
    device_id = int(input("\nChoisissez le numéro du micro : "))
    fs = 44100  # Fréquence d'échantillonnage

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(ROOT_DIR, "data", "enregistrements", f"{nom}", f"{timestamp}")
    os.makedirs(path, exist_ok=True)


    mode = input("\nChoisissez le mode : \n(1) Durée fixe (en minutes) \n(2) Jusqu'à appui sur Entrée\nChoix : ")
    duration = None
    if mode == "1":
        duration = float(input("Durée en minutes : ")) * 60

    print("\nPréparez-vous...")
    for i in range(5, 0, -1):
        time.sleep(0.5)
        print(f"Lancement dans {i}...", end="\r")
        time.sleep(0.5)
    print("\n🔴 ENREGISTREMENT EN COURS... (Appuyez sur ENTRÉE pour stopper)")

    recorded_chunks = []
    stop_recording = False

    def wait_for_input():
        nonlocal stop_recording
        input()
        stop_recording = True


    if mode == "2" or mode == "1":
        thread = threading.Thread(target=wait_for_input)
        thread.start()

    start_time = time.time()


    with sd.InputStream(samplerate=fs, device=device_id, channels=1) as stream:
        while not stop_recording:
            chunk, overflowed = stream.read(fs)  # Lit 1 seconde à la fois
            recorded_chunks.append(chunk)

            if duration and (time.time() - start_time) >= duration:
                break

    print("🟢 Enregistrement terminé.")

    full_audio = np.concatenate(recorded_chunks, axis=0)

    # 5mins chacune
    segment_length_sec = 300
    samples_per_segment = segment_length_sec * fs

    total_samples = len(full_audio)
    num_segments = int(np.ceil(total_samples / samples_per_segment))

    for i in range(num_segments):
        start = i * samples_per_segment
        end = min((i + 1) * samples_per_segment, total_samples)
        segment_data = full_audio[start:end]

        filename = os.path.join(path, f"segment_{i + 1:02d}.wav")
        write(filename, fs, segment_data)
        print(f"Sauvegardé : {filename}")


if __name__ == "__main__":
    try:
        record_audio()
    except KeyboardInterrupt:
        print("\nInterrompu par l'utilisateur.")