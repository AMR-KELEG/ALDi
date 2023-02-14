import re
from arabic_tokenizer import tokenizer


def preprocess(text):
    # Only keep Arabic literals?
    text = re.sub(r"[\u0640\u064b-\u0652]", "", text)
    text = re.sub(r"[^\u0621-\u064a]", " ", text)
    return " ".join(text.split())


def preprocess_comparison(text):
    text = preprocess(text)

    text = re.sub("ى\b", "ي", text)

    # Normalize Hamzat
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)

    # Remove whitespaces
    return re.sub(r"\s", "", text)


def tokenize_text(text):
    """Tokenize a string based on separator regexps."""
    # The more accurate tokenizer is really slow!
    # tokens = [t.value for t in tokenizer.tokenize(text) if t.value.strip()]

    # TODO: Use a better tokenizer
    tokens = text.split()
    return tokens
