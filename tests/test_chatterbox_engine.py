import numpy as np
import pytest

from voicestudio.audio import load_wav
from voicestudio.tts.chatterbox import ChatterboxTTS


class FakeTensor:
    """Imite les méthodes de torch.Tensor utilisées par le moteur."""

    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeModel:
    sr = 24000

    def __init__(self):
        self.calls = []

    def generate(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return FakeTensor(0.2 * np.ones((1, self.sr), dtype=np.float32))  # 1 s par morceau


@pytest.fixture
def engine():
    e = ChatterboxTTS()
    e._tts = FakeModel()  # évite le chargement du vrai modèle
    return e


def test_synthesize_chunks_and_saves_pcm(tmp_path, engine):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"x")
    text = ("Une phrase. " * 40).strip()  # > 250 caractères => plusieurs morceaux
    out = engine.synthesize(text, tmp_path / "o.mp3", lang="FR-fr", reference=ref, cfg_weight=0.3)

    calls = engine._tts.calls
    assert len(calls) >= 2
    for _, kw in calls:
        assert kw["language_id"] == "fr" and kw["audio_prompt_path"] == str(ref)
        assert kw["cfg_weight"] == 0.3 and kw["exaggeration"] == 0.5
    assert out.suffix == ".wav"  # l'extension est forcée
    audio, sr = load_wav(out)
    n = len(calls)
    assert sr == 24000
    assert abs(len(audio) / sr - (n * 1.0 + (n - 1) * 0.25)) < 0.05  # pauses entre morceaux


def test_synthesize_validation(tmp_path, engine):
    with pytest.raises(ValueError, match="Langue"):
        engine.synthesize("salut", tmp_path / "o.wav", lang="xx")
    with pytest.raises(ValueError, match="vide"):
        engine.synthesize("  ", tmp_path / "o.wav")
    with pytest.raises(FileNotFoundError):
        engine.synthesize("salut", tmp_path / "o.wav", reference=tmp_path / "absent.wav")
