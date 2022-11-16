import os
import pandas as pd
from pathlib import Path


def prepare_test_split(BASE_DIR, lang):
    # lang is either "MSA" or "DA"
    subtask_number = BASE_DIR[-2]
    BASE_DIR = f"{BASE_DIR}{lang}"
    OUTPUT_DIR = "data/NADI_mapped/"
    samples_df = pd.read_csv(
        str(Path(BASE_DIR, f"{lang}_test_unlabeled.tsv")), sep="\t"
    )
    tweets = samples_df["#2_tweet"].tolist()

    with open(str(Path(BASE_DIR, f"subtask1{subtask_number}_GOLD.txt")), "r") as f:
        labels = [l.strip() for l in f]

    assert len(tweets) == len(labels)

    EG_tweets = [tweet for tweet, label in zip(tweets, labels) if label == "Egypt"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(str(Path(OUTPUT_DIR, f"test_{lang}.txt")), "w") as f:
        for tweet in EG_tweets:
            f.write(tweet.strip() + "\n")


if __name__ == "__main__":
    prepare_test_split("data/NADI2021/NADI2021_TEST.1.0/Subtask_1.1+2.1_", "MSA")
    prepare_test_split("data/NADI2021/NADI2021_TEST.1.0/Subtask_1.2+2.2_", "DA")
