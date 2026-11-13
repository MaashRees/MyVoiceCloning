"""Synthèse Edge-TTS : brouillon rapide, SANS clonage.

Attention : le texte est envoyé au service en ligne de Microsoft, via une API
non officielle qui peut changer sans préavis. À utiliser pour relire un script,
pas pour la version finale avec votre voix.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

#: (langue, genre) -> voix Edge
VOICES = {
    ("fr", "femme"): "fr-FR-VivienneMultilingualNeural",
    ("fr", "homme"): "fr-FR-RemyMultilingualNeural",
    ("en", "femme"): "en-US-AnaNeural",
    ("en", "homme"): "en-US-GuyNeural",
    ("es", "femme"): "es-ES-ElviraNeural",
    ("es", "homme"): "es-ES-AlvaroNeural",
}


def pick_voice(lang: str, voice: str | None) -> str:
    """``voice`` peut être « femme », « homme » ou le nom exact d'une voix Edge."""
    if voice and "-" in voice:  # nom complet, ex. fr-FR-RemyMultilingualNeural
        return voice
    genre = (voice or "homme").lower()
    try:
        return VOICES[(lang.lower()[:2], genre)]
    except KeyError:
        raise ValueError(
            f"Aucune voix Edge pour lang={lang!r}, voix={voice!r}. Disponibles : {sorted(VOICES)}"
        ) from None


def _run(coro):
    """Exécute une coroutine, même si une boucle asyncio tourne déjà (Gradio, Jupyter...)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    result: dict = {}

    def target() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # relayé au thread appelant
            result["error"] = exc

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


class EdgeTTS:
    name = "edge"
    sends_data_off_machine = True
    supports_cloning = False
    supports_conversion = False

    def synthesize(
        self,
        text: str,
        output: Path,
        *,
        lang: str = "fr",
        voice: str | None = None,
        reference: Path | None = None,
    ) -> Path:
        if not text or not text.strip():
            raise ValueError("Texte vide : rien à synthétiser.")
        import edge_tts

        output = Path(output).with_suffix(".mp3")  # Edge produit toujours du MP3
        output.parent.mkdir(parents=True, exist_ok=True)
        chosen = pick_voice(lang, voice)
        try:
            _run(edge_tts.Communicate(text, chosen).save(str(output)))
        except BaseException:
            output.unlink(missing_ok=True)  # pas de fichier vide laissé derrière
            raise
        if not output.is_file() or output.stat().st_size == 0:
            output.unlink(missing_ok=True)
            raise RuntimeError(f"Edge-TTS n'a produit aucun audio ({chosen}).")
        return output
