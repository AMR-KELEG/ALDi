#!/usr/bin/env python
# coding: utf-8

import os
import editdistance
import pandas as pd
from tqdm import tqdm
from data_preparation_utils import preprocess, preprocess_comparison, dump_file

tqdm.pandas()


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
