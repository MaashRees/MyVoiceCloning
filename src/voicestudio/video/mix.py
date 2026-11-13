"""Ajoute une voix off à une vidéo avec ffmpeg (la vidéo n'est pas ré-encodée)."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: Volume cible usuel des plateformes de vidéo courte, modifiable.
DEFAULT_LUFS = -14.0


@dataclass(frozen=True)
class MediaInfo:
    duration_s: float
    has_audio: bool


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise RuntimeError(
            f"{tool} est introuvable. Installez FFmpeg (https://ffmpeg.org) et ajoutez-le au PATH."
        )
    return path


def probe(path: str | Path) -> MediaInfo:
    result = subprocess.run(
        [
            _require("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Fichier illisible ({path}) : {result.stderr.strip()}")
    info = json.loads(result.stdout)
    duration = float(info.get("format", {}).get("duration") or 0.0)
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    return MediaInfo(duration, has_audio)


def build_command(
    video: Path,
    voice: Path,
    output: Path,
    *,
    video_info: MediaInfo,
    ambient: float = 0.0,
    delay_s: float = 0.0,
    lufs: float | None = DEFAULT_LUFS,
) -> list[str]:
    """Construit la commande ffmpeg.

    ``ambient`` : volume du son d'origine conservé sous la voix (0 = supprimé, 0.25 = discret).
    ``delay_s`` : retard de la voix par rapport au début de la vidéo.
    ``lufs``    : volume final (None = pas de normalisation).
    """
    if not 0.0 <= ambient <= 1.0:
        raise ValueError("ambient doit être entre 0 et 1.")
    if delay_s < 0:
        raise ValueError("delay_s ne peut pas être négatif.")
    ms = int(round(delay_s * 1000))
    voice_chain = f"[1:a]adelay={ms}|{ms},apad[v]" if ms else "[1:a]apad[v]"
    use_ambient = ambient > 0 and video_info.has_audio
    if use_ambient:
        graph = [
            f"[0:a]volume={ambient}[amb]",
            voice_chain,
            "[amb][v]amix=inputs=2:duration=longest:normalize=0[mix]",
        ]
        last = "mix"
    else:
        graph = [voice_chain]
        last = "v"
    if lufs is not None:
        graph.append(f"[{last}]loudnorm=I={lufs}:TP=-1.5:LRA=11[out]")
    else:
        graph.append(f"[{last}]anull[out]")
    return [
        _require("ffmpeg"),
        "-y",
        "-i",
        str(video),
        "-i",
        str(voice),
        "-filter_complex",
        ";".join(graph),
        "-map",
        "0:v:0",
        "-map",
        "[out]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",  # la voix est prolongée par du silence : on borne à la durée de la vidéo
        f"{video_info.duration_s:.3f}",
        "-movflags",
        "+faststart",
        str(output),
    ]


def add_voiceover(
    video: str | Path,
    voice: str | Path,
    output: str | Path,
    *,
    ambient: float = 0.0,
    delay_s: float = 0.0,
    lufs: float | None = DEFAULT_LUFS,
) -> Path:
    video, voice, output = Path(video), Path(voice), Path(output)
    for p in (video, voice):
        if not p.is_file():
            raise FileNotFoundError(f"Fichier introuvable : {p}")
    info = probe(video)
    if info.duration_s <= 0:
        raise RuntimeError(f"Durée de la vidéo illisible : {video}")
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_command(
        video, voice, output, video_info=info, ambient=ambient, delay_s=delay_s, lufs=lufs
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg a échoué :\n{result.stderr.strip()[-800:]}")
    return output
