import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from voicestudio.audio import record as rec

SR = 16000


class FakeStream:
    """Simule PortAudio : appelle le callback toutes les 10 ms avec une sinusoïde int16."""

    def __init__(self, samplerate, device, channels, dtype, callback):
        assert dtype == "int16" and channels == 1
        self.callback, self.n = callback, samplerate // 100
        self._stop = threading.Event()

    def __enter__(self):
        def run():
            t = 0
            while not self._stop.is_set():
                x = (3000 * np.sin(2 * np.pi * 220 * (t + np.arange(self.n)) / SR)).astype(np.int16)
                self.callback(x.reshape(-1, 1), self.n, None, None)
                t += self.n
                time.sleep(0.01)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.thread.join()


@pytest.fixture
def fake_sd(monkeypatch):
    devices = [
        {"name": "Sortie", "max_input_channels": 0, "default_samplerate": 48000},
        {"name": "Micro USB", "max_input_channels": 1, "default_samplerate": SR},
    ]
    sd = SimpleNamespace(
        query_devices=lambda device=None, kind=None: devices if device is None else devices[1],
        InputStream=FakeStream,
    )
    monkeypatch.setattr(rec, "_sd", lambda: sd)
    return sd


def test_list_input_devices_skips_outputs(fake_sd):
    devices = rec.list_input_devices()
    assert [d.name for d in devices] == ["Micro USB"]
    assert devices[0].index == 1


def test_record_fixed_duration_returns_int16(fake_sd):
    levels = []
    audio, sr = rec.record(duration_s=0.5, device=1, on_level=levels.append)
    assert sr == SR and audio.dtype == np.int16
    assert 0.3 * SR <= len(audio) <= 0.8 * SR
    assert levels and max(levels) < 0  # dBFS


def test_record_stops_on_event(fake_sd):
    stop = threading.Event()
    threading.Timer(0.3, stop.set).start()
    started = time.monotonic()
    audio, _ = rec.record(device=1, stop_event=stop)
    assert time.monotonic() - started < 2
    assert len(audio) > 0
