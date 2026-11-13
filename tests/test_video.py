import shutil
import subprocess

import pytest

from voicestudio.video import add_voiceover, probe
from voicestudio.video.mix import MediaInfo, build_command

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg absent")


def _run(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


@pytest.fixture
def media(tmp_path):
    video_audio = tmp_path / "va.mp4"
    _run("-f", "lavfi", "-i", "testsrc=d=6:s=160x120:r=10",
         "-f", "lavfi", "-i", "sine=f=300:d=6", "-shortest", "-pix_fmt", "yuv420p",
         str(video_audio))  # fmt: skip
    silent = tmp_path / "v.mp4"
    _run("-f", "lavfi", "-i", "testsrc=d=6:s=160x120:r=10", "-pix_fmt", "yuv420p", str(silent))
    voice = tmp_path / "voice.wav"
    _run("-f", "lavfi", "-i", "sine=f=500:d=3", str(voice))
    return video_audio, silent, voice


def test_probe(media):
    with_audio, silent, _ = media
    assert probe(with_audio).has_audio and probe(silent).has_audio is False
    assert probe(with_audio).duration_s == pytest.approx(6, abs=0.3)


def test_voiceover_keeps_video_length_and_adds_audio(tmp_path, media):
    _, silent, voice = media
    out = add_voiceover(silent, voice, tmp_path / "out" / "final.mp4", delay_s=1.0)
    info = probe(out)
    assert info.has_audio
    assert info.duration_s == pytest.approx(6, abs=0.4)  # voix 3 s mais vidéo 6 s


def test_voiceover_with_ambient_sound(tmp_path, media):
    with_audio, _, voice = media
    out = add_voiceover(with_audio, voice, tmp_path / "final.mp4", ambient=0.3, lufs=None)
    assert probe(out).has_audio and probe(out).duration_s == pytest.approx(6, abs=0.4)


def test_ffmpeg_failure_is_reported_and_cleaned(tmp_path, media):
    _, silent, _ = media
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"pas de l'audio")
    out = tmp_path / "final.mp4"
    with pytest.raises(RuntimeError, match="ffmpeg|illisible"):
        add_voiceover(silent, bad, out)
    assert not out.exists()


def test_build_command_validation(tmp_path):
    info = MediaInfo(5.0, True)
    with pytest.raises(ValueError):
        build_command(tmp_path / "a", tmp_path / "b", tmp_path / "c", video_info=info, ambient=2)
    with pytest.raises(ValueError):
        build_command(tmp_path / "a", tmp_path / "b", tmp_path / "c", video_info=info, delay_s=-1)
