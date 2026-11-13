"""Profils de voix : ``data/voices/<nom>/ref_<ambiance>.wav``."""

from __future__ import annotations

from pathlib import Path

from voicestudio.config import Settings, get_settings

_DEFAULT_LABEL = "neutre"


def list_voices(settings: Settings | None = None) -> dict[str, list[str]]:
    """Retourne ``{nom: [ambiances]}`` pour chaque profil existant."""
    root = (settings or get_settings()).voices_dir
    if not root.is_dir():
        return {}
    voices = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        labels = [f.stem.removeprefix("ref_") for f in sorted(d.glob("ref_*.wav"))]
        if labels:
            voices[d.name] = labels
    return voices


def resolve_reference(
    voice: str, label: str | None = None, settings: Settings | None = None
) -> Path:
    """Trouve l'extrait de référence d'un profil (ambiance « neutre » par défaut)."""
    root = (settings or get_settings()).voices_dir / voice
    if label:
        path = root / f"ref_{label}.wav"
        if not path.is_file():
            raise FileNotFoundError(f"Extrait introuvable : {path}")
        return path
    default = root / f"ref_{_DEFAULT_LABEL}.wav"
    if default.is_file():
        return default
    found = sorted(root.glob("ref_*.wav")) if root.is_dir() else []
    if not found:
        raise FileNotFoundError(
            f"Aucun extrait pour la voix « {voice} » dans {root}. "
            f"Créez-en un avec : voicestudio record-ref {voice}"
        )
    return found[0]
