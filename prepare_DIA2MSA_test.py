#!/usr/bin/env python
# coding: utf-8

import os
import editdistance
import pandas as pd
from tqdm import tqdm
from data_preparation_utils import preprocess, normalize_arabic_text, dump_file
from pathlib import Path
import matplotlib.pyplot as plt

tqdm.pandas()


def load_words_list(dialect):
    assert dialect in ["Egyptian", "Gulf", "Levantine", "Maghrebi"]
    words_list_filename = str(Path("data", "DIA2MSA", f"{dialect}StrongWords.txt"))
    with open(words_list_filename, "r") as f:
        WORDS = set([l.strip() for l in f])
    return WORDS


def main():
    os.makedirs("data/DIA2MSA_mapped/", exist_ok=True)
    df = pd.read_excel("data/DIA2MSA/EGY2MSA.xls")

    print(df.shape[0])
    EGYPTIAN_WORDS_LIST = load_words_list("Egyptian")
    # Only keep samples of high confidence
    df = df[df["egytomsa:confidence"] > 0.99].copy()
    print(df.shape[0])
    # Drop samples having any of the distinctive terms in their MSA translation
    df = df[
        ~df["msa"].apply(
            lambda s: any([w in EGYPTIAN_WORDS_LIST for w in str(s).split()])
        )
    ]
    print(df.shape[0])

    df["msa"] = df["msa"].apply(preprocess)
    df["tweet"] = df["tweet"].apply(preprocess)
    df["msa_len"] = df["msa"].apply(lambda s: len(s))
    df["tweet_len"] = df["tweet"].apply(lambda s: len(s))
    df["distance"] = df.progress_apply(
        lambda row: editdistance.distance(
            normalize_arabic_text(row["msa"]), normalize_arabic_text(row["tweet"])
        ),
        axis=1,
    )

    df["distance"].plot.hist()
    plt.show()

    df["distance_percentage"] = df.apply(
        lambda row: row["distance"] / row["tweet_len"], axis=1
    )
    df["distance_percentage"].plot.hist(bins=[v for v in range(0, 2, 0.25)])
    plt.show()

    # Keep only translation of least distance (can be more noisy?)
    little_df = df[(df["distance"] >= 5) & (df["distance"] <= 15)].sample(
        n=1000, random_state=42
    )
    most_df = df[(df["distance"] >= 40)].sample(n=1000, random_state=42)

    BASEDIR = "DIA2MSA_mapped"
    for dialect, column in zip(["DA", "MSA"], ["tweet", "msa"]):
        dump_file(
            dialectness_level="little",
            dialect=dialect,
            samples=little_df[column],
            BASEDIR=BASEDIR,
        )
        dump_file(
            dialectness_level="most",
            dialect=dialect,
            samples=most_df[column],
            BASEDIR=BASEDIR,
        )


if __name__ == "__main__":
    main()
