"""Interface commune des moteurs de synthèse vocale.

Tous les moteurs (Edge, Chatterbox local, serveur distant sur Hugging Face...)
se branchent derrière cette interface : le reste du projet ignore où le calcul
a lieu.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class TTSBackend(Protocol):
    name: str
    #: True si le texte ou la voix de référence quitte la machine.
    sends_data_off_machine: bool
    #: True si le moteur sait imiter une voix à partir d'un extrait.
    supports_cloning: bool

    def synthesize(
        self,
        text: str,
        output: Path,
        *,
        lang: str = "fr",
        voice: str | None = None,
        reference: Path | None = None,
    ) -> Path:
        """Écrit l'audio dans ``output`` (ou à côté, si l'extension doit changer) et
        retourne le chemin réel du fichier."""
        ...
