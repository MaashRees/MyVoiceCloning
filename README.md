# 🎙️ MyVoiceCloning — Synthèse vocale personnalisée

Projet de **clonage de voix en local** : à partir de vos propres enregistrements, préparer un jeu de données vocal propre puis générer de la parole (TTS) avec votre voix. Tout tourne sur votre machine, aucune donnée vocale n'est envoyée à un service tiers dans le mode local.

## État d'avancement

| Étape | Statut |
| :-- | :-- |
| Enregistrement guidé et segmentation du jeu de données | ✅ Disponible |
| Normalisation des fichiers audio (WAV PCM 16 bits) | ✅ Disponible |
| Transcription automatique pour produire `metadata.csv` | ✅ Disponible (Google Speech Recognition ou Vosk hors ligne) |
| Entraînement / inférence du modèle de clonage | 🚧 À venir (dépendances de la stack Coqui TTS déjà listées dans `requirements.txt`) |

## Structure

```
src/
├── recorder/
│   ├── recorder.py        # Enregistrement micro, découpage en segments, dossier par session
│   └── test_mic.py        # Test rapide du micro
└── transcribe/
    ├── convertToPCM.py    # Conversion des WAV en PCM 16 bits
    └── from_dataset.py    # Transcription → metadata.csv (SpeechRecognition ou Vosk)
data/                      # Créé à l'exécution : enregistrements/, modèles Vosk
requirements.txt
```

## Installation

**Prérequis** : Python 3.10 ou 3.11 (pas 3.12, conflit avec la bibliothèque TTS) et [FFmpeg](https://ffmpeg.org/).

```bash
git clone https://github.com/MaashRees/MyVoiceCloning.git
cd MyVoiceCloning
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

> Le fichier `requirements.txt` est encodé en UTF-16. Si `pip` refuse de le lire, convertissez-le d'abord en UTF-8.

## Utilisation

1. **Enregistrer sa voix** : lancer `src/recorder/recorder.py`, choisir le micro et le mode de capture (durée fixe ou arrêt manuel avec Entrée). Les segments sont rangés dans `data/enregistrements/<session>/`.
2. **Normaliser l'audio** : `src/transcribe/convertToPCM.py` convertit les fichiers du dossier en WAV PCM 16 bits.
3. **Transcrire** : `src/transcribe/from_dataset.py` produit un fichier `metadata.csv` (séparateur `|`) :
   - `transcribe_SpeechRecognition` utilise la reconnaissance Google (en ligne, français) ;
   - `transcribe_with_vosk` fonctionne **entièrement hors ligne** avec un modèle téléchargé depuis [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) et placé dans `data/vosk/models`.

## Licence

Voir [LICENSE](LICENSE).
