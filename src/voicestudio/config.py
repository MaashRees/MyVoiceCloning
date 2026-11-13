"""Configuration centralisée, lue depuis l'environnement ou un fichier ``.env``.

Rien n'est évalué à l'import : un module peut être importé sans que la
configuration soit complète. Chaque fonctionnalité vérifie ce dont elle a
besoin au moment de s'en servir.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VOICESTUDIO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Dossier où sont rangés voix, enregistrements et sorties (ignoré par git).
    data_dir: Path = Path("data")

    # --- Transcription (faster-whisper) ---
    whisper_model: str = "small"
    whisper_device: str = "auto"  # auto | cpu | cuda
    whisper_compute: str = "auto"  # auto | int8 | float16 | ...
    vocab_file: Path | None = None  # mots-clés pour guider Whisper (optionnel)

    # --- Traduction (API compatible OpenAI : OpenRouter, serveur perso, ...) ---
    translate_base_url: str | None = None
    translate_api_key: str | None = None
    translate_model: str | None = None

    # --- Synthèse vocale ---
    tts_backend: str = "edge"

    @property
    def voices_dir(self) -> Path:
        return self.data_dir / "voices"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"


@lru_cache
def get_settings() -> Settings:
    return Settings()
