"""Client du serveur GPU distant (voir ``voicestudio serve``).

Le même serveur tourne sur Google Colab, un Space Hugging Face ou un VPS GPU : le
client ne change pas, seule l'URL change. L'extrait de référence et le texte sont
envoyés à ce serveur.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from voicestudio.config import Settings, get_settings


def _parse_auth(value: str | None) -> tuple[str, str] | None:
    if not value:
        return None
    user, sep, password = value.partition(":")
    if not sep or not user:
        raise ValueError("VOICESTUDIO_REMOTE_AUTH doit avoir la forme « utilisateur:motdepasse ».")
    return user, password


class RemoteTTS:
    name = "remote"
    sends_data_off_machine = True
    supports_cloning = True
    supports_conversion = True

    def __init__(self, settings: Settings | None = None, client=None):
        self.settings = settings or get_settings()
        self._client = client

    def _get_client(self):
        if self._client is None:
            s = self.settings
            if not s.remote_url:
                raise RuntimeError(
                    "VOICESTUDIO_REMOTE_URL n'est pas défini : lancez le serveur "
                    "(notebook Colab, Space Hugging Face ou `voicestudio serve`) et collez son URL."
                )
            try:
                from gradio_client import Client
            except ImportError as exc:
                raise RuntimeError(
                    "gradio_client est absent : `pip install 'voicestudio[remote]'`."
                ) from exc
            kwargs = {}
            if s.remote_hf_token:
                kwargs["token"] = s.remote_hf_token
            if auth := _parse_auth(s.remote_auth):
                kwargs["auth"] = auth
            self._client = Client(s.remote_url, verbose=False, **kwargs)
        return self._client

    @staticmethod
    def _file(path: Path | None):
        if path is None:
            return None
        from gradio_client import handle_file

        if not Path(path).is_file():
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        return handle_file(str(path))

    def synthesize(
        self,
        text: str,
        output: Path,
        *,
        lang: str = "fr",
        voice: str | None = None,
        reference: Path | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ) -> Path:
        if not text or not text.strip():
            raise ValueError("Texte vide : rien à synthétiser.")
        client = self._get_client()
        try:
            result = client.predict(
                text,
                lang,
                self._file(reference),
                exaggeration,
                cfg_weight,
                api_name="/synthesize",
            )
        except Exception as exc:
            raise RuntimeError(f"Échec de l'appel au serveur distant : {exc}") from exc
        return self._save(result, output)

    def convert(self, audio: Path, target_voice: Path, output: Path) -> Path:
        client = self._get_client()
        try:
            result = client.predict(
                self._file(audio), self._file(target_voice), api_name="/convert"
            )
        except Exception as exc:
            raise RuntimeError(f"Échec de l'appel au serveur distant : {exc}") from exc
        return self._save(result, output)

    @staticmethod
    def _save(result, output: Path) -> Path:
        src = Path(result[0] if isinstance(result, (tuple, list)) else result)
        if not src.is_file():
            raise RuntimeError("Le serveur n'a renvoyé aucun fichier audio.")
        output = Path(output).with_suffix(src.suffix or ".wav")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, output)
        return output
