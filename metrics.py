import requests
import settings
from abc import ABC, abstractmethod
from typing import Any
import utils
import editdistance
import pickle
from utils import tokenize_text
from pathlib import Path
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    AutoModelForTokenClassification,
)
from camel_tools.dialectid import DIDModel6


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

    def backtranslate(self, arabic_text):
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


class LexiconOverlapMetric(DialectnessLevelMetric):
    def __init__(self, lexicon_source):
        # Make sure the lexicons are generated using "form_msa_lexicon.py"
        assert lexicon_source in ["UN", "opensubtitles"]
        lexicon_path = str(
            Path("data/MSA_raw_corpora/", f"lexicon_{lexicon_source}.pkl")
        )
        with open(lexicon_path, "rb") as f:
            self.LEXICON = pickle.load(f)

    def compute_dialectness_score(self, text):
        """Compute the percentage of tokens that can not be found in an MSA lexicon.

        Args:
            text: The text to compute the dialectness score for.

        Returns:
            A dialectness score in range [0, 1] based on number of tokens not in lexicon.
        """
        tokens = tokenize_text(text)

        # TODO: The filtering step below should be done on building the LEXICON!
        # Ignore words occuring once
        return 1 - (
            len([t for t in tokens if t in self.LEXICON and self.LEXICON[t] > 1])
            / len(tokens)
        )


class RegressionBERTMetric:
    def __init__(self, model_path, model_name="UBC-NLP/MARBERT"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_path, num_labels=1
        )

    def compute_dialectness_score(self, text):
        # TODO: Validate the max_length!
        return (
            self.model(
                **self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512
                )
            )
            .logits.squeeze(-1)[0]
            .tolist()
        )

    # TODO: Add another function to perform batch predictions


class LIBERTMetric:
    def __init__(self, model_path, model_name="UBC-NLP/MARBERT"):
        TAGS = ["ambiguous", "lang1", "lang2", "mixed", "ne", "other"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path, num_labels=len(TAGS)
        )

        self.INDECIES_TO_TAGS = {i: tag for i, tag in enumerate(TAGS)}

    def compute_dialectness_score(self, text):
        logits = self.model(
            **self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        ).logits

        # Ignore the labels for [CLS] and [SEP]
        subwords_labels = logits.argmax(axis=-1).numpy()[0][1:-1]
        tokens = utils.tokenize_text(text)
        subwords = [
            self.tokenizer.tokenize(token)
            for token in tokens
            if self.tokenizer.tokenize(token)
        ]
        n_subwords = [len(l) for l in subwords]
        first_subword_indecies = [sum(n_subwords[0:i]) for i in range(len(n_subwords))]

        # TODO: Validate the condition!
        tokens_labels = [
            subwords_labels[index] for index in first_subword_indecies if index < 510
        ]
        tokens_tags = [self.INDECIES_TO_TAGS[l] for l in tokens_labels]

        # Compute the CMI (Code Mixing Index)
        # Ignore: "ambiguous", "ne" (named entity), "other" (emojis, ..)
        n_msa_tokens = sum([t == "lang1" for t in tokens_tags])
        n_da_tokens = sum([t in ["lang2", "mixed"] for t in tokens_tags])

        if n_msa_tokens + n_da_tokens != 0:
            return n_da_tokens / (n_msa_tokens + n_da_tokens)
        else:
            return 0


class DIMetric:
    def __init__(self):
        self.di_model = DIDModel6.pretrained()

    def compute_dialectness_score(self, text):
        dialect = self.di_model.predict([text])[0].top
        # TODO: Use the scores of the DI model?
        return 0 if dialect == "MSA" else 1
