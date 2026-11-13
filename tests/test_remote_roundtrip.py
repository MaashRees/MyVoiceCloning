"""Aller-retour réel client <-> serveur Gradio, avec un faux moteur (sans GPU)."""

import numpy as np
import pytest

pytest.importorskip("gradio")
pytest.importorskip("gradio_client")

from voicestudio.audio import load_wav, save_wav  # noqa: E402
from voicestudio.config import Settings  # noqa: E402
from voicestudio.server import build_app  # noqa: E402
from voicestudio.tts.remote import RemoteTTS, _parse_auth  # noqa: E402


class FakeEngine:
    def __init__(self):
        self.seen = {}

    def synthesize(self, text, output, *, lang, reference, exaggeration, cfg_weight):
        self.seen["synth"] = (text, lang, reference is not None, exaggeration, cfg_weight)
        return save_wav(output, 0.2 * np.ones(8000), 16000)

    def convert(self, audio, target_voice, output):
        self.seen["convert"] = (audio.name, target_voice.name)
        return save_wav(output, 0.1 * np.ones(4000), 16000)


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    engine = FakeEngine()
    demo = build_app(engine)
    _, url, _ = demo.launch(server_name="127.0.0.1", prevent_thread_lock=True, quiet=True)
    yield engine, url
    demo.close()


def test_synthesize_roundtrip(server, tmp_path):
    engine, url = server
    ref = save_wav(tmp_path / "ref.wav", 0.2 * np.ones(16000), 16000)
    client = RemoteTTS(Settings(_env_file=None, remote_url=url))
    out = client.synthesize(
        "Bonjour", tmp_path / "res.mp3", lang="fr", reference=ref, exaggeration=0.7, cfg_weight=0.3
    )
    assert out.suffix == ".wav"
    audio, sr = load_wav(out)
    assert sr == 16000 and len(audio) == 8000
    assert engine.seen["synth"] == ("Bonjour", "fr", True, 0.7, 0.3)


def test_convert_roundtrip(server, tmp_path):
    engine, url = server
    a = save_wav(tmp_path / "prise.wav", 0.2 * np.ones(16000), 16000)
    r = save_wav(tmp_path / "cible.wav", 0.2 * np.ones(16000), 16000)
    out = RemoteTTS(Settings(_env_file=None, remote_url=url)).convert(a, r, tmp_path / "c.wav")
    assert len(load_wav(out)[0]) == 4000
    assert engine.seen["convert"] == ("prise.wav", "cible.wav")


def test_remote_configuration_errors():
    with pytest.raises(RuntimeError, match="REMOTE_URL"):
        RemoteTTS(Settings(_env_file=None)).synthesize("salut", "x.wav")
    assert _parse_auth("moi:secret:avec:deux-points") == ("moi", "secret:avec:deux-points")
    assert _parse_auth(None) is None
    with pytest.raises(ValueError):
        _parse_auth("sans-separateur")
