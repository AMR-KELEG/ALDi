#!/usr/bin/env python
# coding: utf-8

import os
import re
import editdistance
import pandas as pd
from tqdm import tqdm
from pathlib import Path

tqdm.pandas()


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


def dump_file(dialectness_level, dialect, samples):
    filepath = Path("data", "DIA2MSA_mapped", f"{dialectness_level}_{dialect}.txt")
    with open(str(filepath), "w") as f:
        for s in samples:
            f.write(s + "\n")


def main():
    os.makedirs("data/DIA2MSA_mapped/", exist_ok=True)
    df = pd.read_excel("data/DIA2MSA/EGY2MSA.xls")
    # Only keep samples of high confidence
    df = df[df["egytomsa:confidence"] > 0.99].copy()
    df["msa"] = df["msa"].apply(preprocess)
    df["tweet"] = df["tweet"].apply(preprocess)
    df["msa_len"] = df["msa"].apply(lambda s: len(s))
    df["tweet_len"] = df["tweet"].apply(lambda s: len(s))
    df["distance"] = df.progress_apply(
        lambda row: editdistance.distance(
            preprocess_comparison(row["msa"]), preprocess_comparison(row["tweet"])
        ),
        axis=1,
    )

    little_df = df[(df["distance"] >= 5) & (df["distance"] <= 15)].sample(
        n=1000, random_state=42
    )
    most_df = df[(df["distance"] >= 40)].sample(n=1000, random_state=42)

    for dialect, column in zip(["DA", "MSA"], ["tweet", "msa"]):
        dump_file(
            dialectness_level="little", dialect=dialect, samples=little_df[column]
        )
        dump_file(dialectness_level="most", dialect=dialect, samples=most_df[column])


if __name__ == "__main__":
    main()
