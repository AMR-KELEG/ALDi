import re


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
