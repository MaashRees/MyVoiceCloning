import os
import speech_recognition as sr
import csv
from pyprojroot import here

ROOT_DIR = here()
r = sr.Recognizer()


def transcribe_SpeechRecognition(folder_path):
    output_csv = os.path.join(folder_path, "metadata.csv")
    files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]

    with open(output_csv, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            print(f"Transcription de {filename}...")

            with sr.AudioFile(file_path) as source:
                audio_data = r.record(source)
                try:
                    text = r.recognize_google(audio_data, language="fr-FR")
                    writer.writerow([filename, text])
                    print(f"Réussi : {text[:50]}...")
                except Exception as e:
                    print(f"Erreur sur {filename}: {e}")
                    writer.writerow([filename, "ERREUR_TRANSCRIPTION"])


if __name__ == "__main__":
    mon_dossier = os.path.join(ROOT_DIR, "data", "enregistrements", "20260109_001656")
    transcribe_SpeechRecognition(mon_dossier)