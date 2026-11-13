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
    # edge (brouillon, sans clonage) | chatterbox (local, GPU) | remote (serveur GPU distant)
    tts_backend: str = "edge"
    chatterbox_device: str = "auto"  # auto | cuda | cpu | mps
    chatterbox_model: str = "v3"  # modèle multilingue : v3 (recommandé) ou v2

    # --- Serveur GPU distant (Colab, Space Hugging Face, VPS GPU) ---
    remote_url: str | None = None  # ex. https://xxxx.gradio.live ou owner/space
    remote_auth: str | None = None  # « utilisateur:motdepasse » si le serveur est protégé
    remote_hf_token: str | None = None  # pour un Space privé

    @property
    def voices_dir(self) -> Path:
        return self.data_dir / "voices"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"


@lru_cache
def get_settings() -> Settings:
    return Settings()
