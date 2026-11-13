# MyVoiceCloning 🎙️ : voix off personnalisée

Outil pour enregistrer sa voix, transcrire, traduire et générer des voix off (vidéos de voyage,
scénarios de shooting...).

> **Statut : refonte en cours (étape 1 sur 3).**
> Les fondations sont prêtes et testées. Le **clonage de voix lui-même n'est pas encore branché** :
> il arrive à l'étape 2 (voir la feuille de route).

## Ce qui fonctionne aujourd'hui

| Commande | Rôle | Local ? |
|---|---|---|
| `voicestudio devices` | Liste les micros | oui |
| `voicestudio record-ref moi` | Enregistre un extrait de référence (PCM 16 bits), le nettoie et vérifie sa qualité | oui |
| `voicestudio check-ref fichier.wav` | Contrôle qualité d'un extrait existant (durée, saturation, niveau, Bluetooth dégradé…) | oui |
| `voicestudio transcribe video.mp4 --srt sous-titres.srt` | Transcription Whisper et sous-titres | **oui** (CPU ou GPU) |
| `voicestudio translate "texte" --from fr --to en` | Traduction par un LLM | **non** : le texte part vers l'API que vous configurez |
| `voicestudio say "texte"` | Voix de brouillon **sans clonage** (Edge-TTS) | **non** : le texte part chez Microsoft |

Seuls la capture audio, l'analyse et Whisper sont réellement locaux.

## Installation

Python 3.10 ou plus récent (3.13 inclus : la contrainte « pas de 3.12 » de l'ancienne version
venait de la bibliothèque Coqui `TTS`, abandonnée et retirée du projet).

```bash
git clone https://github.com/MaashRees/MyVoiceCloning.git
cd MyVoiceCloning
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -e ".[record,asr]"     # micro + transcription locale
cp .env.example .env               # puis adaptez-le
```

- Micro sous Ubuntu : `sudo apt install libportaudio2`.
- Windows : rien de plus à installer pour le micro.
- Sans les extras, l'outil reste utilisable pour `translate`, `say` et `check-ref`.

## Utilisation

```bash
voicestudio devices
voicestudio record-ref moi --label neutre --device 1     # Entrée pour arrêter
voicestudio check-ref data/voices/moi/ref_neutre.wav
voicestudio transcribe mon_clip.mp4 --lang fr --srt mon_clip.srt
voicestudio say "Bonjour à tous !" --voice femme -o data/output/test.mp3
```

Les fichiers sont rangés dans `data/` (ignoré par git).

### Conseils pour un bon extrait de référence
10 à 30 secondes de phrases continues, pièce calme, micro à 15-20 cm, ton naturel.
Avec un micro sans fil (DJI, etc.), utilisez le récepteur USB et **pas** le mode Bluetooth mains
libres : il enregistre en basse qualité et `record-ref` vous en avertira.

## Feuille de route

1. ✅ **Fondations** : paquet Python, configuration, enregistrement, contrôle qualité,
   Whisper, traduction, sous-titres, tests, CI.
2. ⏳ **Voix off clonée** : moteur Chatterbox derrière l'interface `tts`, serveur Gradio à
   déployer sur un Space Hugging Face ou un GPU loué (aucun GPU local nécessaire), commande
   `voiceover video.mp4 script.txt` (mixage ffmpeg, volume adapté à Instagram).
3. ⏳ **Confort créatif** : voice-to-voice (garder votre intonation), profils de voix,
   versions multilingues avec votre voix.
4. 💤 **Plus tard** : traducteur temps réel pour réunions.

## Développement

```bash
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

## Éthique et consentement

Ne clonez la voix d'une personne qu'avec son accord explicite. Pour les contenus publiés,
vérifiez les règles d'étiquetage des contenus générés par IA de chaque plateforme.

## Licence

MIT
