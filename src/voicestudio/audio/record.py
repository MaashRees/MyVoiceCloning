"""Enregistrement micro (sounddevice). L'import de PortAudio est paresseux."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    channels: int
    default_samplerate: int


def _sd():
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:  # OSError : PortAudio absent
        raise RuntimeError(
            "sounddevice est indisponible. Installez-le avec `pip install 'voicestudio[record]'` "
            "(sous Ubuntu, il faut aussi `sudo apt install libportaudio2`)."
        ) from exc
    return sd


def list_input_devices() -> list[InputDevice]:
    """Périphériques disposant d'au moins un canal d'entrée."""
    sd = _sd()
    return [
        InputDevice(i, d["name"], int(d["max_input_channels"]), int(d["default_samplerate"]))
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]


def record(
    duration_s: float | None = None,
    device: int | None = None,
    samplerate: int | None = None,
    on_level: Callable[[float], None] | None = None,
    stop_event: threading.Event | None = None,
) -> tuple[np.ndarray, int]:
    """Enregistre en mono, directement en PCM 16 bits.

    S'arrête quand ``duration_s`` est écoulée, quand ``stop_event`` est levé
    ou sur Ctrl+C (l'audio déjà capté est conservé).
    ``on_level`` reçoit le niveau crête de chaque bloc en dBFS.
    """
    sd = _sd()
    if samplerate is None:
        samplerate = int(sd.query_devices(device, "input")["default_samplerate"])

    stop = stop_event or threading.Event()
    blocks: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status):  # appelé par PortAudio
        block = indata[:, 0].copy()
        blocks.put(block)
        if on_level is not None:
            peak = float(np.max(np.abs(block))) / 32768.0
            on_level(20 * np.log10(max(peak, 1e-6)))

    chunks: list[np.ndarray] = []
    started = time.monotonic()
    try:
        with sd.InputStream(
            samplerate=samplerate,
            device=device,
            channels=1,
            dtype="int16",
            callback=callback,
        ):
            while not stop.is_set():
                if duration_s is not None and time.monotonic() - started >= duration_s:
                    break
                try:
                    chunks.append(blocks.get(timeout=0.1))
                except queue.Empty:
                    pass
    except KeyboardInterrupt:
        pass

    while not blocks.empty():
        chunks.append(blocks.get_nowait())
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    return audio, samplerate


def stop_on_enter() -> threading.Event:
    """Retourne un événement levé quand l'utilisateur appuie sur Entrée."""
    event = threading.Event()

    def wait() -> None:
        try:
            input()
        except EOFError:
            pass
        event.set()

    threading.Thread(target=wait, daemon=True).start()
    return event
