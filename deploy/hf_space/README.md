---
title: voicestudio
emoji: 🎙️
sdk: gradio
app_file: app.py
pinned: false
---

# Déploiement sur Hugging Face

1. Créez un **Space privé**, SDK Gradio, matériel **ZeroGPU** (abonnement PRO) ou un GPU payant.
2. Déposez-y `app.py`, `requirements.txt` et ce `README.md` (l'en-tête ci-dessus est requis).
3. Dans `.env` sur votre PC :
   ```
   VOICESTUDIO_TTS_BACKEND=remote
   VOICESTUDIO_REMOTE_URL=votre-compte/votre-space
   VOICESTUDIO_REMOTE_HF_TOKEN=hf_...
   ```
4. Testez : `voicestudio say "Bonjour" --voice moi`.

Les quotas et tarifs ZeroGPU changent : vérifiez-les sur la page officielle avant de payer.
