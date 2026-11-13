"""Moteur de clonage Chatterbox Multilingual (V3), exécuté sur la machine courante.

Prévu pour une machine avec GPU (Colab, Space Hugging Face, VPS GPU). Sur CPU ça
fonctionne mais reste très lent. Le projet Chatterbox est archivé depuis août 2026
(licence MIT) : il reste utilisable, sans mises à jour.

Installation (la version V3 n'est PAS sur PyPI, seulement sur GitHub ; commit figé) :

    pip install "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git@ab55923c27757c90cfcc407baa57dc25a7d25217"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from voicestudio.audio.processing import save_wav
from voicestudio.tts.text import split_text

#: Codes de langue du modèle multilingue (README de Chatterbox).
SUPPORTED_LANGUAGES = frozenset(
    "ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh".split()
)
_PAUSE_S = 0.25  # silence inséré entre deux morceaux


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch est absent : ce moteur doit tourner sur la machine GPU. "
            "Installez Chatterbox (voir l'en-tête de voicestudio/tts/chatterbox.py)."
        ) from exc
    return torch


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    torch = _torch()
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _to_numpy(wav) -> np.ndarray:
    return wav.detach().cpu().numpy().reshape(-1).astype(np.float32)


class ChatterboxTTS:
    name = "chatterbox"
    sends_data_off_machine = False
    supports_cloning = True
    supports_conversion = True

    def __init__(self, device: str = "auto", model: str = "v3"):
        self.device_request = device
        self.model_version = model
        self._tts = None
        self._vc = None
        self.device: str | None = None

    # -- chargement paresseux ------------------------------------------------
    def load(self) -> None:
        """Charge le modèle de synthèse (téléchargement à la première fois)."""
        if self._tts is not None:
            return
        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        except ImportError as exc:
            raise RuntimeError(
                "Chatterbox n'est pas installé (voir l'en-tête de voicestudio/tts/chatterbox.py)."
            ) from exc
        self.device = pick_device(self.device_request)
        if self.device == "cpu":
            print("Attention : pas de GPU détecté, la synthèse sera très lente.")
        self._tts = ChatterboxMultilingualTTS.from_pretrained(
            device=self.device, t3_model=self.model_version
        )

    def _load_vc(self):
        if self._vc is None:
            try:
                from chatterbox.vc import ChatterboxVC
            except ImportError as exc:
                raise RuntimeError("Chatterbox n'est pas installé.") from exc
            self.device = self.device or pick_device(self.device_request)
            self._vc = ChatterboxVC.from_pretrained(self.device)
        return self._vc

    # -- synthèse ------------------------------------------------------------
    def synthesize(
        self,
        text: str,
        output: Path,
        *,
        lang: str = "fr",
        voice: str | None = None,
        reference: Path | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ) -> Path:
        """Synthétise ``text`` avec la voix de ``reference`` (10 à 30 s idéalement)."""
        lang = lang.lower()[:2]
        if lang not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Langue non gérée : {lang!r}")
        chunks = split_text(text)
        if not chunks:
            raise ValueError("Texte vide : rien à synthétiser.")
        if reference is not None and not Path(reference).is_file():
            raise FileNotFoundError(f"Extrait de référence introuvable : {reference}")
        self.load()

        parts: list[np.ndarray] = []
        for chunk in chunks:
            wav = self._tts.generate(
                chunk,
                language_id=lang,
                audio_prompt_path=str(reference) if reference else None,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
            parts.append(_to_numpy(wav))
            parts.append(np.zeros(int(_PAUSE_S * self._tts.sr), dtype=np.float32))
        audio = np.concatenate(parts[:-1])  # pas de silence final
        return save_wav(Path(output).with_suffix(".wav"), audio, int(self._tts.sr))

    def convert(self, audio: Path, target_voice: Path, output: Path) -> Path:
        """Voice-to-voice : garde l'intonation de ``audio``, prend le timbre de ``target_voice``."""
        for p in (audio, target_voice):
            if not Path(p).is_file():
                raise FileNotFoundError(f"Fichier introuvable : {p}")
        vc = self._load_vc()
        wav = vc.generate(audio=str(audio), target_voice_path=str(target_voice))
        return save_wav(Path(output).with_suffix(".wav"), _to_numpy(wav), int(vc.sr))
