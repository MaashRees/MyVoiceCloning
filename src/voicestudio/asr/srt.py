"""Export des segments en sous-titres SRT."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from voicestudio.asr.whisper import Segment


def format_timestamp(seconds: float) -> str:
    ms_total = int(round(max(seconds, 0.0) * 1000))
    h, rest = divmod(ms_total, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(segments: Iterable[Segment]) -> str:
    blocks = []
    for i, seg in enumerate(segments, start=1):
        blocks.append(
            f"{i}\n{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n{seg.text}\n"
        )
    return "\n".join(blocks)


def write_srt(segments: Iterable[Segment], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_srt(segments), encoding="utf-8")
    return path
