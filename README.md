# MyVoiceCloning 🎙️ : voix off personnalisée

Outil pour enregistrer sa voix, transcrire, traduire et générer des voix off (vidéos de voyage,
scénarios de shooting...).

> **Statut : refonte en cours (étape 2 sur 3).**
> Le code du clonage de voix et du mixage vidéo est écrit et testé **avec un faux moteur**.
> **Le vrai modèle (Chatterbox) n'a pas encore été exécuté** : je n'avais pas de GPU pour le
> tester. Le premier essai sur Colab ou Hugging Face est l'étape de validation qui reste.

## Ce qui fonctionne aujourd'hui

| Commande | Rôle | Local ? |
|---|---|---|
| `voicestudio devices` | Liste les micros | oui |
| `voicestudio record-ref moi` | Enregistre un extrait de référence (PCM 16 bits), le nettoie et vérifie sa qualité | oui |
| `voicestudio check-ref fichier.wav` | Contrôle qualité d'un extrait existant (durée, saturation, niveau, Bluetooth dégradé…) | oui |
| `voicestudio transcribe video.mp4 --srt sous-titres.srt` | Transcription Whisper et sous-titres | **oui** (CPU ou GPU) |
| `voicestudio translate "texte" --from fr --to en` | Traduction par un LLM | **non** : le texte part vers l'API que vous configurez |
| `voicestudio say "texte"` | Voix de brouillon **sans clonage** (Edge-TTS) | **non** : le texte part chez Microsoft |
| `voicestudio say "texte" --voice moi --backend remote` | **Voix clonée** sur un serveur GPU | **non** : texte et extrait de voix partent vers votre serveur |
| `voicestudio voiceover clip.mp4 script.txt --voice moi` | Génère la voix et l'ajoute à la vidéo (volume normalisé à -14 LUFS, vidéo non ré-encodée) | dépend du moteur |
| `voicestudio convert prise.wav --voice moi` | Voice-to-voice : garde votre intonation, prend votre timbre | idem |
| `voicestudio voices` | Liste vos profils de voix | oui |
| `voicestudio serve` | Lance le serveur de clonage (sur la machine GPU) | — |

Seuls la capture audio, l'analyse, Whisper et le mixage vidéo sont toujours locaux.

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

## Cloner sa voix : où fait-on tourner le modèle ?

Le PC n'a pas besoin de GPU : le modèle tourne sur un serveur, votre PC l'appelle.

| Où | Comment |
|---|---|
| **Google Colab** (GPU gratuit, session temporaire) | Ouvrez `notebooks/colab_server.ipynb`, exécutez les cellules, copiez l'URL et le mot de passe dans `.env`. |
| **Space Hugging Face** (privé) | Voir `deploy/hf_space/README.md`. |
| **Serveur Ubuntu avec GPU** | `pip install -e ".[server]"` + Chatterbox, puis `voicestudio serve --auth moi:motdepasse`. |

Puis, dans `.env` : `VOICESTUDIO_TTS_BACKEND=remote` et les variables `VOICESTUDIO_REMOTE_*`
(voir `.env.example`), et `pip install -e ".[remote]"` sur le PC.

**Modèle utilisé : Chatterbox Multilingual V3** (MIT, français inclus). **Attention : le projet
Chatterbox est archivé depuis août 2026**, sans mises à jour. Il reste utilisable ; le moteur est
isolé derrière l'interface `tts` pour pouvoir en changer. La version V3 n'est pas sur PyPI : on
l'installe depuis GitHub, à un commit figé.

## Utilisation

```bash
voicestudio devices
voicestudio record-ref moi --label neutre --device 1     # Entrée pour arrêter
voicestudio check-ref data/voices/moi/ref_neutre.wav
voicestudio transcribe mon_clip.mp4 --lang fr --srt mon_clip.srt
voicestudio say "Bonjour à tous !" --voice femme -o data/output/test.mp3   # brouillon Edge

# Avec votre voix clonée (serveur GPU configuré) :
voicestudio say "Bienvenue à Lisbonne." --voice moi --backend remote
voicestudio voiceover clip.mp4 script.txt --voice moi --backend remote --ambient 0.2
voicestudio convert ma_prise.wav --voice moi --backend remote
```

`--ambient 0.2` garde le son d'origine de la vidéo en fond discret sous la voix (0 = supprimé).

Les fichiers sont rangés dans `data/` (ignoré par git).

### Conseils pour un bon extrait de référence
10 à 30 secondes de phrases continues, pièce calme, micro à 15-20 cm, ton naturel.
Avec un micro sans fil (DJI, etc.), utilisez le récepteur USB et **pas** le mode Bluetooth mains
libres : il enregistre en basse qualité et `record-ref` vous en avertira.

## Feuille de route

1. ✅ **Fondations** : paquet Python, configuration, enregistrement, contrôle qualité,
   Whisper, traduction, sous-titres, tests, CI.
2. 🟡 **Voix off clonée** : code, serveur, client, mixage ffmpeg et notebook Colab écrits ;
   plomberie testée avec un faux moteur. **Reste à valider avec le vrai modèle sur GPU.**
3. ⏳ **Confort créatif** : musique de fond, sous-titres incrustés, versions multilingues
   avec votre voix, second moteur de secours (le projet Chatterbox n'est plus maintenu).
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
