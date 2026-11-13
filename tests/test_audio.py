import numpy as np
import pytest

from voicestudio.audio import (
    analyze,
    load_wav,
    reference_warnings,
    resample,
    save_wav,
    split_on_silence,
    trim_silence,
)
from voicestudio.audio.processing import to_float, to_int16

SR = 16000


def tone(seconds, freq=220.0, amp=0.3, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds, sr=SR):
    return np.zeros(int(seconds * sr), dtype=np.float32)


def test_int16_roundtrip_and_clipping():
    x = np.array([0.0, 0.5, -0.5, 2.0, -2.0], dtype=np.float32)
    out = to_int16(x)
    assert out.dtype == np.int16
    assert out[3] == 32767 and out[4] == -32768
    assert np.allclose(to_float(out)[:3], x[:3], atol=1e-3)


def test_stereo_is_mixed_to_mono():
    stereo = np.stack([tone(0.1), tone(0.1)], axis=1)
    assert to_float(stereo).ndim == 1


def test_resample_length():
    out = resample(tone(1.0, sr=44100), 44100, 16000)
    assert abs(len(out) - 16000) <= 2


def test_save_wav_is_pcm16_and_does_not_touch_source(tmp_path):
    path = tmp_path / "sub" / "a.wav"
    save_wav(path, tone(0.5), SR)
    import soundfile as sf

    assert sf.info(str(path)).subtype == "PCM_16"
    data, sr = load_wav(path)
    assert sr == SR and abs(len(data) - SR // 2) <= 1


def test_trim_silence_keeps_speech_and_margin():
    x = np.concatenate([silence(1), tone(1), silence(1)])
    out = trim_silence(x, SR, pad_s=0.1)
    assert 1.0 <= len(out) / SR <= 1.4
    assert len(trim_silence(silence(2), SR)) == 0


def test_split_on_silence_cuts_phrases():
    x = np.concatenate([tone(3), silence(1), tone(3), silence(1), tone(3)])
    parts = split_on_silence(x, SR, min_s=2, max_s=10)
    assert len(parts) == 3
    assert all(2.5 <= len(p) / SR <= 4.5 for p in parts)


def test_split_on_silence_enforces_max_length():
    parts = split_on_silence(tone(30), SR, min_s=2, max_s=10)
    assert len(parts) >= 3
    assert all(len(p) / SR <= 10.01 for p in parts)


def test_good_reference_has_no_warning():
    assert reference_warnings(analyze(tone(15, amp=0.2), SR)) == []


def test_reference_warnings_cover_known_problems():
    short = reference_warnings(analyze(tone(2), SR))
    assert any("court" in w for w in short)
    clipped = reference_warnings(analyze(tone(15, amp=1.5).clip(-1, 1), SR))
    assert any("saturé" in w for w in clipped)
    quiet = reference_warnings(analyze(tone(15, amp=0.001), SR))
    assert any("faible" in w for w in quiet)
    narrow = reference_warnings(analyze(tone(15, amp=0.2, sr=8000), 8000))
    assert any("Bluetooth" in w for w in narrow)
    mostly_silent = reference_warnings(analyze(np.concatenate([tone(5), silence(15)]), SR))
    assert any("silencieuse" in w for w in mostly_silent)


@pytest.mark.parametrize("seconds", [0.0, 0.01])
def test_analyze_tiny_inputs_do_not_crash(seconds):
    analyze(tone(seconds), SR)
