import re
from pathlib import Path


def preprocess(text):
    # Only keep Arabic literals?
    text = re.sub(r"[\u0640\u064b-\u0652]", "", text)
    text = re.sub(r"[^\u0621-\u064a]", " ", text)
    return " ".join(text.split())


def preprocess_comparison(text):
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى$", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    return re.sub(r"\s", "", text)


def dump_file(dialectness_level, dialect, samples, BASEDIR):
    filepath = Path("data", BASEDIR, f"{dialectness_level}_{dialect}.txt")
    with open(str(filepath), "w") as f:
        for s in samples:
            f.write(s + "\n")
