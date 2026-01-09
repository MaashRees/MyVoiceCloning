import os
import speech_recognition as sr
import csv

from pyprojroot import here
from vosk import Model, KaldiRecognizer
import soundfile as sf
import wave
import json

ROOT_DIR = here()
r = sr.Recognizer()
vosk_site = r"https://alphacephei.com/vosk/models"
VOST_DIR = os.path.join(ROOT_DIR, "data", "vosk", "models")


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


def transcribe_with_vosk(folder_path, model_path):
    if not os.path.exists(model_path):
        print(f"❌ Modèle introuvable dans {model_path}. Télécharge-le sur le site de Vosk : {vosk_site}!")
        return

    model = Model(model_path)
    output_csv = os.path.join(folder_path, "metadata_vosk.csv")
    files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]

    print(f"🎙️ Début de la transcription locale de {len(files)} fichiers...")

    with open(output_csv, mode='w', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter='|')

        for filename in files:
            file_path = os.path.join(folder_path, filename)

            # --- CONVERSION ---
            temp_path = "temp_vosk.wav"
            data, fs = sf.read(file_path)
            sf.write(temp_path, data, fs, subtype='PCM_16')

            # --- TRANSCRIPTION ---
            wf = wave.open(temp_path, "rb")
            rec = KaldiRecognizer(model, wf.getframerate())
            rec.SetWords(True)

            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    pass

            res = json.loads(rec.FinalResult())
            final_text = res.get("text", "")

            writer.writerow([filename, final_text])
            print(f"✅ {filename} : {final_text[:60]}...")

            wf.close()
            if os.path.exists(temp_path): os.remove(temp_path)

    print(f"\n✨ Terminé ! Ton fichier est prêt : {output_csv}")

if __name__ == "__main__":
    mon_dossier = os.path.join(ROOT_DIR, "data", "enregistrements", "20260109_001656")
    model_name = "vosk-model-small-fr-0.22"
    model_path = os.path.join(VOST_DIR, model_name)
    # transcribe_SpeechRecognition(mon_dossier)
    transcribe_with_vosk(mon_dossier, model_path)