from voicestudio.config import Settings, get_settings
from voicestudio.tts.base import TTSBackend

BACKENDS = ("edge", "chatterbox", "remote")


def get_backend(name: str, settings: Settings | None = None) -> TTSBackend:
    """Retourne le moteur demandé (import paresseux : torch n'est chargé qu'au besoin)."""
    settings = settings or get_settings()
    name = name.lower()
    if name == "edge":
        from voicestudio.tts.edge import EdgeTTS

        return EdgeTTS()
    if name == "chatterbox":
        from voicestudio.tts.chatterbox import ChatterboxTTS

        return ChatterboxTTS(device=settings.chatterbox_device, model=settings.chatterbox_model)
    if name == "remote":
        from voicestudio.tts.remote import RemoteTTS

        return RemoteTTS(settings)
    raise ValueError(f"Moteur TTS inconnu : {name!r} (disponibles : {', '.join(BACKENDS)})")


__all__ = ["BACKENDS", "TTSBackend", "get_backend"]
