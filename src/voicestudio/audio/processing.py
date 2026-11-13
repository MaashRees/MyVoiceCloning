"""Traitements audio purs (numpy) : conversion, découpe, contrôle qualité.

Aucune dépendance au micro ni à un modèle : tout est testable sans matériel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

_EPS = 1e-10


# --------------------------------------------------------------------------- #
# Conversion et fichiers
# --------------------------------------------------------------------------- #
def to_float(x: np.ndarray) -> np.ndarray:
    """Retourne un signal float32 mono dans [-1, 1]."""
    x = np.asarray(x)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    else:
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def to_int16(x: np.ndarray) -> np.ndarray:
    """Convertit en PCM 16 bits (avec écrêtage), format attendu par la plupart des outils."""
    x = to_float(x)
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return to_float(x)
    g = int(np.gcd(sr_in, sr_out))
    return resample_poly(to_float(x), sr_out // g, sr_in // g).astype(np.float32)


def save_wav(path: str | Path, data: np.ndarray, sr: int) -> Path:
    """Écrit un WAV mono PCM 16 bits. Ne modifie jamais un fichier source."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), to_int16(data), sr, subtype="PCM_16")
    return path


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return to_float(data), int(sr)


# --------------------------------------------------------------------------- #
# Mesures
# --------------------------------------------------------------------------- #
def peak_dbfs(x: np.ndarray) -> float:
    return float(20 * np.log10(max(float(np.max(np.abs(x))) if len(x) else 0.0, _EPS)))


def rms_dbfs(x: np.ndarray) -> float:
    if len(x) == 0:
        return -120.0
    return float(20 * np.log10(max(float(np.sqrt(np.mean(np.square(x)))), _EPS)))


def frame_db(x: np.ndarray, sr: int, frame_s: float = 0.03) -> np.ndarray:
    """Niveau RMS (dBFS) par trame de ``frame_s`` secondes."""
    n = max(1, int(sr * frame_s))
    count = len(x) // n
    if count == 0:
        return np.array([rms_dbfs(x)])
    frames = x[: count * n].reshape(count, n)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    return 20 * np.log10(np.maximum(rms, _EPS))


# --------------------------------------------------------------------------- #
# Silences et découpe
# --------------------------------------------------------------------------- #
def trim_silence(
    x: np.ndarray, sr: int, threshold_db: float = -45.0, pad_s: float = 0.15
) -> np.ndarray:
    """Retire le silence au début et à la fin (en gardant un peu de marge)."""
    x = to_float(x)
    frame_s = 0.03
    n = int(sr * frame_s)
    active = np.flatnonzero(frame_db(x, sr, frame_s) > threshold_db)
    if len(active) == 0:
        return x[:0]
    pad = int(pad_s * sr)
    start = max(0, int(active[0]) * n - pad)
    end = min(len(x), (int(active[-1]) + 1) * n + pad)
    return x[start:end]


def split_on_silence(
    x: np.ndarray,
    sr: int,
    *,
    min_s: float = 2.0,
    max_s: float = 15.0,
    silence_s: float = 0.35,
    threshold_db: float = -40.0,
) -> list[np.ndarray]:
    """Découpe un long enregistrement en phrases de ``min_s`` à ``max_s`` secondes.

    Coupe au milieu des silences d'au moins ``silence_s``. Les morceaux trop
    longs sont recoupés à leur point le plus faible. Utile pour préparer un
    jeu de données (fine-tuning) ; inutile pour un clonage zero-shot.
    """
    x = to_float(x)
    frame_s = 0.03
    n = int(sr * frame_s)
    db = frame_db(x, sr, frame_s)
    silent = db <= threshold_db

    # Points de coupe = milieu de chaque silence assez long.
    cuts = [0]
    i = 0
    min_frames = max(1, int(silence_s / frame_s))
    while i < len(silent):
        if silent[i]:
            j = i
            while j < len(silent) and silent[j]:
                j += 1
            if j - i >= min_frames:
                cuts.append(((i + j) // 2) * n)
            i = j
        else:
            i += 1
    cuts.append(len(x))
    cuts = sorted(set(min(c, len(x)) for c in cuts))

    max_n, min_n = int(max_s * sr), int(min_s * sr)
    pieces: list[tuple[int, int]] = []
    start = cuts[0]
    for end in cuts[1:]:
        if end - start >= min_n or end == len(x):
            pieces.append((start, end))
            start = end
    if start < len(x):  # reste trop court : on le recolle au précédent
        if pieces:
            pieces[-1] = (pieces[-1][0], len(x))
        else:
            pieces.append((start, len(x)))

    out: list[np.ndarray] = []
    for a, b in pieces:
        while b - a > max_n:
            lo, hi = a + int(max_n * 0.6), a + max_n
            window = db[lo // n : max(lo // n + 1, hi // n)]
            cut = lo + int(np.argmin(window)) * n if len(window) else hi
            out.append(x[a:cut])
            a = cut
        if b - a > 0:
            out.append(x[a:b])
    return [seg for seg in out if len(trim_silence(seg, sr, threshold_db)) > 0]


# --------------------------------------------------------------------------- #
# Contrôle qualité d'un extrait de référence
# --------------------------------------------------------------------------- #
@dataclass
class AudioReport:
    duration_s: float
    sample_rate: int
    peak_dbfs: float
    speech_rms_dbfs: float
    clipping_ratio: float
    silence_ratio: float


def analyze(x: np.ndarray, sr: int) -> AudioReport:
    x = to_float(x)
    db = frame_db(x, sr)
    speech = db > -40.0
    # Niveau moyen mesuré uniquement sur les parties où l'on parle.
    n = int(sr * 0.03)
    if speech.any() and len(x) >= n:
        mask = np.repeat(speech, n)[: len(x)]
        speech_rms = rms_dbfs(x[: len(mask)][mask])
    else:
        speech_rms = rms_dbfs(x)
    return AudioReport(
        duration_s=len(x) / sr if sr else 0.0,
        sample_rate=sr,
        peak_dbfs=peak_dbfs(x),
        speech_rms_dbfs=speech_rms,
        clipping_ratio=float(np.mean(np.abs(x) >= 0.999)) if len(x) else 0.0,
        silence_ratio=float(1 - speech.mean()) if len(speech) else 1.0,
    )


def reference_warnings(report: AudioReport) -> list[str]:
    """Problèmes susceptibles de dégrader un clone de voix."""
    w: list[str] = []
    if report.duration_s < 6:
        w.append(f"Extrait trop court ({report.duration_s:.1f} s) : visez 10 à 30 s.")
    elif report.duration_s > 60:
        w.append(
            f"Extrait long ({report.duration_s:.0f} s) : 10 à 30 s suffisent et suivent mieux."
        )
    if report.sample_rate < 16000:
        w.append(
            f"Fréquence d'échantillonnage faible ({report.sample_rate} Hz) : "
            "typique d'un micro Bluetooth en mode mains libres. Utilisez un mode haute qualité."
        )
    if report.clipping_ratio > 0.001 or report.peak_dbfs > -0.5:
        w.append("Signal saturé : baissez le gain du micro et réenregistrez.")
    if report.speech_rms_dbfs < -30:
        w.append(
            f"Voix trop faible ({report.speech_rms_dbfs:.0f} dBFS) : rapprochez-vous du micro."
        )
    if report.silence_ratio > 0.5:
        w.append("Plus de la moitié de l'extrait est silencieuse : coupez les blancs.")
    return w
