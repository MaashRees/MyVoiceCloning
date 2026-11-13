from voicestudio.tts.base import TTSBackend


def get_backend(name: str) -> TTSBackend:
    """Retourne le moteur de synthèse demandé."""
    name = name.lower()
    if name == "edge":
        from voicestudio.tts.edge import EdgeTTS

        return EdgeTTS()
    if name in {"chatterbox", "remote"}:
        raise NotImplementedError(
            f"Le moteur « {name} » (clonage de voix) arrive à l'étape 2 de la refonte."
        )
    raise ValueError(f"Moteur TTS inconnu : {name!r} (disponible : edge)")


__all__ = ["TTSBackend", "get_backend"]
