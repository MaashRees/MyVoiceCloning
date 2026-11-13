"""Transcription locale avec faster-whisper (CPU ou GPU, détecté automatiquement)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from voicestudio.config import Settings, get_settings

# Whisper n'accepte qu'environ 224 tokens de contexte initial.
_MAX_PROMPT_CHARS = 600


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str
    segments: list[Segment]


def load_vocab_prompt(path: str | Path | None) -> str | None:
    """Construit le prompt initial de Whisper à partir d'un fichier de mots-clés.

    Une ligne = un mot ou plusieurs séparés par des virgules ; ``#`` = commentaire.
    Retourne ``None`` si le fichier est absent ou vide.
    """
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    words: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words.extend(w.strip() for w in line.split(",") if w.strip())
    if not words:
        return None
    return f"Vocabulaire : {', '.join(words)}."[:_MAX_PROMPT_CHARS]


def resolve_device(device: str = "auto", compute: str = "auto") -> tuple[str, str]:
    """Choisit le périphérique et le type de calcul (``auto`` : GPU si présent, sinon CPU)."""
    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


class Transcriber:
    """Charge le modèle Whisper à la première utilisation, pas à l'import."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper n'est pas installé : `pip install 'voicestudio[asr]'`."
                ) from exc
            device, compute = resolve_device(
                self.settings.whisper_device, self.settings.whisper_compute
            )
            self._model = WhisperModel(
                self.settings.whisper_model, device=device, compute_type=compute
            )
        return self._model

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        use_vocab: bool = False,
    ) -> Transcript:
        """Transcrit un fichier audio. ``language=None`` : détection automatique."""
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Fichier audio introuvable : {audio_path}")
        prompt = load_vocab_prompt(self.settings.vocab_file) if use_vocab else None
        raw_segments, info = self._load().transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,
            initial_prompt=prompt,
        )
        segments = [Segment(s.start, s.end, s.text.strip()) for s in raw_segments]
        text = " ".join(s.text for s in segments).strip()
        return Transcript(text=text, language=info.language, segments=segments)
