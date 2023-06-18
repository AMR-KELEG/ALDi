import os
import tarfile
import pandas as pd
from pathlib import Path
from urllib.request import urlretrieve

DATASET_DIR = Path("data/Bible/")
DATASET_URL = (
    "https://alt.qcri.org/resources/mt/arabench/releases/current/AraBench_dataset.tgz"
)
DATASET_FILE = str(Path(DATASET_DIR, DATASET_URL.split("/")[-1]))


def download_arabench():
    os.makedirs(DATASET_DIR, exist_ok=True)
    urlretrieve(DATASET_URL, DATASET_FILE)

    tar = tarfile.open(DATASET_FILE, "r:gz")
    for item in tar:
        tar.extract(item, DATASET_DIR)


def load_file(split, dialect):
    assert dialect in ["tn", "ma", "msa"]

    base_filename = str(Path(DATASET_DIR, f"AraBench_dataset/bible.{split}."))
    filename = base_filename + (
        f"mgr.0.{dialect}.ar" if dialect != "msa" else "msa.0.ms.ar"
    )
    with open(filename, "r") as f:
        return [l.strip() for l in f]


def main():
    download_arabench()

    ma_sentences = load_file("dev", "ma") + load_file("test", "ma")
    tn_sentences = load_file("dev", "tn") + load_file("test", "tn")
    msa_sentences = load_file("dev", "msa") + load_file("test", "msa")

    assert len(ma_sentences) == len(tn_sentences) == len(msa_sentences)

    bible_df = pd.DataFrame(
        {"ma": ma_sentences, "tn": tn_sentences, "msa": msa_sentences}
    )

    # Tunisian Bible has chapters numbers as part of the text (Check dev.0 - lines 0, 17, 32, ... ||| test.0 - lines "1174 - 600").
    # Some verses have ":" appearing at the start of the line (verse?) which is strange.
    # Chapters numbers appear as well (Check dev.0 - line 143, 178).

    # Some lines have footnotes(?) starting with # that are not part of the verse (Check dev.9 - lines 11, ...)

    bible_df.to_csv(str(Path(DATASET_DIR, "bible.tsv")), sep="\t", index=False)


if __name__ == "__main__":
    main()
