"""Point d'entrée d'un Space Hugging Face (SDK Gradio, matériel ZeroGPU ou GPU).

Non testé sur Hugging Face depuis le dépôt : à valider au premier déploiement.
Le Space doit être PRIVÉ (il traite votre voix) ; côté client, renseignez
VOICESTUDIO_REMOTE_URL=<propriétaire>/<space> et VOICESTUDIO_REMOTE_HF_TOKEN.
"""

from voicestudio.server import build_app
from voicestudio.tts.chatterbox import ChatterboxTTS

try:
    import spaces  # présent uniquement sur Hugging Face

    decorate = spaces.GPU(duration=90)
except ImportError:
    decorate = None

engine = ChatterboxTTS(device="cuda")
engine.load()  # chargé au démarrage : ne consomme pas le quota GPU à chaque requête

demo = build_app(engine, decorate=decorate)

if __name__ == "__main__":
    demo.queue().launch()
