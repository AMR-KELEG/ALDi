import requests
import settings
from abc import ABC, abstractmethod
from typing import Any
import utils
import editdistance


class DialectnessLevelMetric(ABC):
    @abstractmethod
    def compute_dialectness_score(self, dialectal_sentence, msa_translation):
        pass


class BackTranslationMetric(DialectnessLevelMetric):
    def __init__(self):
        # API key acquired from https://mt.qcri.org/api/register
        self.API_KEY = settings.SHAHIN_API_KEY

    def find_translation(self, lang_pair: str, domain: str, text: str) -> str:
        """Translate the text using one of QCRI's Shaheen models.

        Args:
            lang_pair: ar-en or en-ar
            domain: The domain of the model from https://mt.qcri.org/api/servers
            text: The text to be translated

        Returns:
            A translation of the text.
        """
        n_retries = 5
        url = f"https://mt.qcri.org/api/v1/translate?key={self.API_KEY}&langpair={lang_pair}&domain={domain}&text={text}"

        while n_retries:
            try:
                response = requests.get(url)
                return response.json()["translatedText"]
            except Exception as e:
                n_retries -= 1

                # Raise an exception if the translation process can not be completed
                if n_retries == 0:
                    raise e

    def backtranslate(self, arabic_text: str) -> dict[str, str | dict[str, Any]]:
        """Backtranslate text using QCRI's Shaheen models.
        Note: The method is slow!

        Args:
            arabic_text: An input text to backtranslate.

        Returns:
            Back translated versions of the input text.
        """
        translated_english_general = self.find_translation(
            lang_pair="ar-en", domain="general-fast", text=arabic_text
        )
        translated_english_dialectal = self.find_translation(
            lang_pair="ar-en", domain="dialectal", text=arabic_text
        )
        translated_arabic_general = self.find_translation(
            lang_pair="en-ar", domain="general", text=translated_english_general
        )
        translated_arabic_dialectal = self.find_translation(
            lang_pair="en-ar", domain="general", text=translated_english_dialectal
        )
        return {
            "arabic": arabic_text,
            "translated_english": {
                "general": translated_english_general,
                "dialectal": translated_english_dialectal,
            },
            "backtranslated_arabic": {
                "general": translated_arabic_general,
                "dialectal": translated_arabic_dialectal,
            },
        }

    def compute_dialectness_score(self, dialectal_sentence, msa_translation=None):
        """Compute the dialectness score based on the lexical overlap between the sentence and its (back)translation(s).

        Args:
            dialectal_sentence: An input sentence.
            msa_translation: A Gold standard MSA translation. Defaults to None.

        Returns:
            A dialectenss score in range [0, 1].
        """

        translations = [
            v
            for v in self.backtranslate(dialectal_sentence)[
                "backtranslated_arabic"
            ].values()
        ]
        if msa_translation:
            translations += [msa_translation]

        preprocessed_dialectal_sentence = utils.preprocess_comparison(
            dialectal_sentence
        )
        preprocessed_translations = [
            utils.preprocess_comparison(sentence) for sentence in translations
        ]

        distances = [
            editdistance.distance(
                preprocessed_dialectal_sentence, preprocessed_translation
            )
            / max(len(preprocessed_dialectal_sentence), len(preprocessed_translation))
            for preprocessed_translation in preprocessed_translations
        ]

        return max(distances)
