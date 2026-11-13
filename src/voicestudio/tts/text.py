"""Découpage d'un script en morceaux adaptés aux modèles de synthèse.

Les modèles de TTS dérivent (répétitions, hallucinations) sur les textes longs :
on les synthétise phrase par phrase puis on recolle l'audio.
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")


def _split_long(sentence: str, max_chars: int) -> list[str]:
    """Coupe une phrase trop longue aux virgules, sinon aux espaces."""
    parts: list[str] = []
    current = ""
    for token in re.split(r"(?<=[,;:])\s+", sentence):
        if current and len(current) + 1 + len(token) > max_chars:
            parts.append(current)
            current = token
        else:
            current = f"{current} {token}".strip()
    if current:
        parts.append(current)
    out: list[str] = []
    for part in parts:
        while len(part) > max_chars:
            cut = part.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            out.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            out.append(part)
    return out


def split_text(text: str, max_chars: int = 250) -> list[str]:
    """Découpe en morceaux d'au plus ``max_chars`` caractères, sans casser les phrases
    quand c'est possible. Les phrases courtes voisines sont regroupées."""
    sentences = [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        pieces = [sentence] if len(sentence) <= max_chars else _split_long(sentence, max_chars)
        for piece in pieces:
            if current and len(current) + 1 + len(piece) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks
