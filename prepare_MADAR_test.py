import os
import pandas as pd
import editdistance
from tqdm import tqdm
from pathlib import Path
from data_preparation_utils import preprocess, normalize_arabic_text, dump_file
import matplotlib.pyplot as plt

tqdm.pandas()


def main():
    os.makedirs("data/MADAR_mapped/", exist_ok=True)
    BASEDIR = (
        "data/MADAR/MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021/MADAR_Corpus/"
    )

    EGY_df = pd.read_csv(str(Path(BASEDIR, "MADAR.corpus.Cairo.tsv")), sep="\t")
    EGY_df["EGY"] = EGY_df["sent"]
    MSA_df = pd.read_csv(str(Path(BASEDIR, "MADAR.corpus.MSA.tsv")), sep="\t")
    MSA_df["MSA"] = MSA_df["sent"]

    df = pd.merge(EGY_df, MSA_df, on="sentID.BTEC")
    df["MSA"] = df["MSA"].apply(preprocess)
    df["EGY"] = df["EGY"].apply(preprocess)
    df["MSA_len"] = df["MSA"].apply(lambda s: len(s))
    df["EGY_len"] = df["EGY"].apply(lambda s: len(s))
    df["distance"] = df.progress_apply(
        lambda row: editdistance.distance(
            normalize_arabic_text(row["MSA"]), normalize_arabic_text(row["EGY"])
        ),
        axis=1,
    )

    df["distance"].plot.hist()
    plt.show()

    df["distance_percentage"] = df.apply(
        lambda row: row["distance"] / row["EGY_len"], axis=1
    )
    df["distance_percentage"].plot.hist(bins=[v for v in range(0, 2, 0.25)])
    plt.show()

    little_df = df[(df["distance"] >= 5) & (df["distance"] <= 15)].sample(
        n=200, random_state=42
    )
    most_df = df[(df["distance"] >= 40)].sample(n=200, random_state=42)

    BASEDIR = "MADAR_mapped/"
    for dialect, column in zip(["DA", "MSA"], ["EGY", "MSA"]):
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
