import shutil
import subprocess

import numpy as np
import pytest
from typer.testing import CliRunner

from voicestudio.audio import save_wav
from voicestudio.cli import app
from voicestudio.config import get_settings
from voicestudio.video import probe

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg absent")


class FakeEngine:
    name = "fake"
    sends_data_off_machine = False
    supports_cloning = True
    supports_conversion = True

    def __init__(self):
        self.calls = []

    def synthesize(self, text, output, **kwargs):
        self.calls.append((text, kwargs))
        return save_wav(output, 0.2 * np.ones(16000 * 2), 16000)

    def convert(self, audio, target_voice, output):
        self.calls.append(("convert", audio.name, target_voice.name))
        return save_wav(output, 0.2 * np.ones(16000), 16000)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICESTUDIO_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    engine = FakeEngine()
    monkeypatch.setattr("voicestudio.tts.get_backend", lambda name, settings=None: engine)
    ref = tmp_path / "data" / "voices" / "moi" / "ref_neutre.wav"
    save_wav(ref, 0.2 * np.ones(16000 * 12), 16000)
    yield tmp_path, engine
    get_settings.cache_clear()


def test_voiceover_end_to_end(env):
    tmp_path, engine = env
    video = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=d=5:s=160x120:r=10",
         "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )  # fmt: skip
    script = tmp_path / "script.txt"
    script.write_text("Bienvenue à Lisbonne.\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["voiceover", str(video), str(script), "--voice", "moi"])
    assert result.exit_code == 0, result.output
    final = tmp_path / "clip_voiceover.mp4"
    assert final.is_file() and probe(final).has_audio
    text, kwargs = engine.calls[0]
    assert text == "Bienvenue à Lisbonne."
    assert kwargs["reference"].name == "ref_neutre.wav" and kwargs["lang"] == "fr"


def test_voices_and_convert(env):
    tmp_path, engine = env
    assert "moi : neutre" in CliRunner().invoke(app, ["voices"]).output
    take = save_wav(tmp_path / "prise.wav", 0.2 * np.ones(16000), 16000)
    result = CliRunner().invoke(app, ["convert", str(take), "--voice", "moi"])
    assert result.exit_code == 0, result.output
    assert engine.calls[0] == ("convert", "prise.wav", "ref_neutre.wav")


def test_unknown_voice_gives_actionable_error(env):
    result = CliRunner().invoke(app, ["say", "salut", "--voice", "inconnu"])
    assert result.exit_code == 1
    assert "record-ref inconnu" in result.output
