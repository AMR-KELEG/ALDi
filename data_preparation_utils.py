import re
from pathlib import Path


def preprocess(text):
    # Remove diactritics
    text = re.sub(r"[\u0640\u064b-\u0652]", "", text)
    # Remove any non-Arabic literal
    text = re.sub(r"[^\u0621-\u064a]", " ", text)
    return " ".join(text.split())


def normalize_arabic_text(text):
    # Normalize Alef
    text = re.sub("[إأٱآا]", "ا", text)
    # Normalize Yaa
    text = re.sub("ى$", "ي", text)
    # Normalize Hamza
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    return re.sub(r"\s", "", text)


def dump_file(dialectness_level, dialect, samples, BASEDIR):
    filepath = Path("data", BASEDIR, f"{dialectness_level}_{dialect}.txt")
    with open(str(filepath), "w") as f:
        for s in samples:
            f.write(s + "\n")
