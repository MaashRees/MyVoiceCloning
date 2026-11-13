"""Serveur Gradio exposant le clonage de voix à distance (``voicestudio serve``).

Même code pour Colab, Hugging Face Spaces et un VPS GPU. Les endpoints
``/synthesize`` et ``/convert`` sont ceux qu'appelle ``tts.remote.RemoteTTS``.
"""

from __future__ import annotations

import itertools
import tempfile
from collections.abc import Callable
from pathlib import Path

LANGUAGES = ["fr", "en", "es", "de", "it", "pt", "nl", "ar", "ja", "ko", "zh"]


def build_app(engine, decorate: Callable | None = None):
    """Construit l'application Gradio autour d'un moteur (``synthesize`` / ``convert``).

    ``decorate`` enveloppe les fonctions de calcul, p. ex. ``spaces.GPU`` sur un Space ZeroGPU.
    """
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("gradio est absent : `pip install 'voicestudio[server]'`.") from exc

    deco = decorate or (lambda fn: fn)
    workdir = Path(tempfile.mkdtemp(prefix="voicestudio_"))

    @deco
    def synthesize(text, lang, reference, exaggeration, cfg_weight):
        if not text or not text.strip():
            raise gr.Error("Texte vide.")
        out = workdir / f"tts_{next(_counter)}.wav"
        path = engine.synthesize(
            text,
            out,
            lang=lang,
            reference=Path(reference) if reference else None,
            exaggeration=float(exaggeration),
            cfg_weight=float(cfg_weight),
        )
        return str(path)

    @deco
    def convert(audio, reference):
        if not audio or not reference:
            raise gr.Error("Il faut un audio source et un extrait de référence.")
        out = workdir / f"vc_{next(_counter)}.wav"
        return str(engine.convert(Path(audio), Path(reference), out))

    with gr.Blocks(title="voicestudio") as demo:
        gr.Markdown(
            "## voicestudio : clonage de voix\n"
            "Ne clonez que votre propre voix ou celle d'une personne qui y a consenti."
        )
        with gr.Tab("Texte vers voix"):
            text = gr.Textbox(label="Texte", lines=4)
            lang = gr.Dropdown(LANGUAGES, value="fr", label="Langue")
            ref = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Votre voix")
            exa = gr.Slider(0.25, 2.0, value=0.5, step=0.05, label="Expressivité")
            cfg = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Rythme (cfg)")
            out = gr.Audio(label="Résultat", type="filepath")
            gr.Button("Générer").click(
                synthesize, [text, lang, ref, exa, cfg], out, api_name="synthesize"
            )
        with gr.Tab("Voix vers voix"):
            src = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Votre prise")
            ref2 = gr.Audio(sources=["upload"], type="filepath", label="Voix cible")
            out2 = gr.Audio(label="Résultat", type="filepath")
            gr.Button("Convertir").click(convert, [src, ref2], out2, api_name="convert")
    return demo


_counter = itertools.count(1)
