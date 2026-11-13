"""Traduction via n'importe quelle API compatible OpenAI (OpenRouter, serveur perso, ...)."""

from __future__ import annotations

from voicestudio.config import Settings, get_settings

LANGUAGE_NAMES = {
    "fr": "French",
    "en": "English",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ar": "Arabic",
}


class TranslationError(RuntimeError):
    """Levée quand la traduction échoue ; on ne renvoie jamais un message d'erreur
    à la place du texte, car il serait ensuite lu à voix haute."""


def language_name(code_or_name: str) -> str:
    return LANGUAGE_NAMES.get(code_or_name.lower(), code_or_name)


def _clean(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'«»":
        text = text[1:-1].strip()
    return text


class Translator:
    def __init__(self, settings: Settings | None = None, client=None):
        self.settings = settings or get_settings()
        self._client = client

    def _get_client(self):
        if self._client is None:
            s = self.settings
            missing = [
                name
                for name, value in (
                    ("VOICESTUDIO_TRANSLATE_BASE_URL", s.translate_base_url),
                    ("VOICESTUDIO_TRANSLATE_API_KEY", s.translate_api_key),
                    ("VOICESTUDIO_TRANSLATE_MODEL", s.translate_model),
                )
                if not value
            ]
            if missing:
                raise TranslationError(
                    "Configuration de traduction incomplète, variables manquantes : "
                    + ", ".join(missing)
                    + " (voir .env.example)."
                )
            from openai import OpenAI

            self._client = OpenAI(base_url=s.translate_base_url, api_key=s.translate_api_key)
        return self._client

    def translate(self, text: str, source: str = "fr", target: str = "en") -> str:
        if not text or not text.strip():
            return ""
        system = (
            "You are a professional interpreter. "
            f"Translate the user's text from {language_name(source)} to {language_name(target)}. "
            "Output ONLY the translation, with no quotes, notes or explanations."
        )
        try:
            response = self._get_client().chat.completions.create(
                model=self.settings.translate_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
            )
        except TranslationError:
            raise
        except Exception as exc:
            raise TranslationError(f"Échec de l'appel au service de traduction : {exc}") from exc
        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            raise TranslationError("Le service de traduction a renvoyé une réponse vide.")
        return _clean(content)
