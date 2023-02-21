"""
The AOC dataset is grouped by HIT (i.e.: 12 annotations for each row).
This script explodes the dataset such that each row is a single annotation.

Since annotators are given the option to provide their information only once,
the script also adds the annotator's information to each row.
"""

import os
import re
import math
import statistics
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from itertools import groupby
from collections import Counter

tqdm.pandas()
from urllib.request import urlretrieve

BASE_DATASET_DIR = "data/AOC"
RAW_DATASET_FILE = str(Path(BASE_DATASET_DIR, "dialectid_data.tsv"))
DATASET_URL = "https://raw.githubusercontent.com/sjeblee/AOC/master/stuff-from-omar/annotated-aoc-data/dialectid_data.tsv"


def download_AOC():
    # Create the dataset directory
    os.makedirs(BASE_DATASET_DIR, exist_ok=True)

    # Retrieve the dataset
    urlretrieve(DATASET_URL, RAW_DATASET_FILE)


def explode_AOC():
    """Explode HIT annotation rows into single annotation ones."""

    df = pd.read_csv(RAW_DATASET_FILE, sep="\t", on_bad_lines="skip", low_memory=False)
    columns = ["Sentence", "DClass", "DLevel", "source", "document", "part", "segment"]
    dfs = []

    for i in range(1, 13):
        columns_names = [
            "HIT_info",
            "HIT-ID",
            "AID",
            "AssignmentID",
            "HITStatus",
            "WorkerIP",
            "WorkerCountry",
            "WorkerCity",
            "country_live",
            "native_arabic",
            "native_dialect",
            "years_speaking",
            "StartTime_str",
            "EndTime_str",
            "WorkTime",
            "comment",
        ]
        columns_names += [f"{c}{i}" for c in columns]
        dfs.append(df[columns_names].copy())
        dfs[-1].columns = [re.sub("\d+$", "", c) for c in columns_names]
        dfs[-1]["segment_order_in_page"] = i

    concat_df = pd.concat(dfs)

    # Infer information about the annotators
    # Steps:
    # 1) Group information by AID
    # 2) Find the most common value for country_live, native_arabic, native_dialect, years_speaking
    # 3) Generate a dictionary for the most common values for each column per annotator
    # 4) Use the dictionary to populate the exploded version of the dataset
    annotator_information_columns = [
        "country_live",
        "native_arabic",
        "native_dialect",
        "years_speaking",
    ]
    annotator_information_dict = {}

    for AID, gdf in concat_df.groupby("AID"):
        annotator_information_dict[AID] = {"# annotations": gdf.shape[0]}
        for col in annotator_information_columns:
            # TODO: Ignore annotators having contradictions in the information?
            # Find the most common value per column
            col_values = [
                v
                for v in gdf[col].tolist()
                if not (v == "prompt" or (type(v) == float and math.isnan(v)))
            ]
            most_common_value = statistics.mode(col_values) if col_values else None
            annotator_information_dict[AID][col] = most_common_value

    # Use the inferred annotator information
    for col in annotator_information_columns:
        concat_df[col] = concat_df.progress_apply(
            lambda row: annotator_information_dict[row["AID"]][col], axis=1
        )

    concat_df["is_control_sentence"] = concat_df["source"].apply(
        lambda s: str(s).endswith("_a")
    )

    return concat_df


def augment_AOC_cols(df):
    df["#_tokens"] = df["Sentence"].apply(lambda s: len(str(s).split()))

    sentences_count = Counter(df["Sentence"])
    df["#_annotations"] = df["Sentence"].apply(lambda s: sentences_count[s])

    # Generate an ID for each sentence to refer to it later
    sentences_id = {s: i for i, (s, c) in enumerate(sentences_count.most_common())}
    df["SID"] = df["Sentence"].apply(lambda s: sentences_id[s])

    # Find the set of sentences marked at least one as junk or without a label
    junk_sentences = set(
        df.loc[df["DLevel"].isin(["junk", "prompt"]), "Sentence"].tolist()
    )
    df["is_junk"] = df["Sentence"].apply(lambda s: s in junk_sentences)

    # Map the categorical dialectness levels to numeric values
    dialectness_level_map = {"most": -3, "mixed": -2, "little": -1, "msa": +2}
    df["dialectness_level"] = df["DLevel"].apply(
        lambda l: dialectness_level_map.get(l, None)
    )

    return df


def group_annotations_by_sentence_id(df):
    """Group annotations into rows (one for each sentence)

    Args:
        df: A dataframe of single annotation per row

    Returns:
        A grouped dataframe of annotations
    """
    # Form the annotations df - grouped by the sentence
    annotations = []
    for sentence, group in groupby(
        [row for i, row in df.iterrows()],
        key=lambda row: row["Sentence"],
    ):
        group_items = list(group)
        group_items = sorted(group_items, key=lambda d: d["AID"])
        dialect_level = [
            str(group_item["DLevel"])
            for group_item in group_items
            if str(group_item["DLevel"]) not in ["nan", "prompt", "junk"]
        ]
        dialect_class = [group_item["DClass"] for group_item in group_items]
        native_dialect = [
            str(group_item["native_dialect"])
            for group_item in group_items
            if str(group_item["native_dialect"]) not in ["nan"]
        ]
        annotator_location = [group_item["country_live"] for group_item in group_items]
        annotator_country_from_IP = [
            group_item["WorkerCountry"] for group_item in group_items
        ]

        dialectness_level = [
            group_item["dialectness_level"] for group_item in group_items
        ]
        # Skip sentences if the dialect of the annotator is unknown!
        if len(native_dialect) != 3 or len(dialect_level) != 3:
            continue

        annotations.append(
            {
                "sentence": sentence,
                "dialect_level": dialect_level,
                "dialect": dialect_class,
                "dialectness_level": dialectness_level,
                "average_dialectness_level": sum(dialectness_level)
                / len(dialectness_level),
                "same_polarity": all([s < 0 for s in dialectness_level])
                or all([s > 0 for s in dialectness_level]),
                "same_label": len(set(dialectness_level)) == 1,
                "annotator_dialect": native_dialect,
                "annotator_location": annotator_location,
                "annotator_country_from_IP": annotator_country_from_IP,
                "source": group_items[0]["source"],
                "document": group_items[0]["document"],
                "annotator_id": [group_item["AID"] for group_item in group_items],
            }
        )

    annotations_df = pd.DataFrame(annotations)
    return annotations_df


if __name__ == "__main__":
    download_AOC()

    df = explode_AOC()
    df.to_csv(str(Path(BASE_DATASET_DIR, "AOC_exploded.tsv")), index=False, sep="\t")

    df = augment_AOC_cols(df)

    # Filter samples keeping only sentences satisfying the following criteria:
    # 1) Sentences longer than 4 tokens and shorter than 27 tokens (based on previous research)
    # 2) Sentences with 3 annotations
    # 3) Sentences that are not mark as junk by any of the 3 annotators
    # 4) Sentences that are not extracted from articles' bodies
    single_sentence_comment_df = (
        df[
            (df["#_tokens"] >= 5)
            & (df["#_tokens"] <= 27)
            & (df["#_annotations"] == 3)
            & (~df["is_junk"])
            & (df["source"].apply(lambda s: not str(s).endswith("_a")))
        ]
        .copy()
        .reset_index(drop=True)
    )

    annotations_df = group_annotations_by_sentence_id(single_sentence_comment_df)
    annotations_df.to_csv(
        str(Path(BASE_DATASET_DIR, "AOC_aggregated.tsv")), index=False, sep="\t"
    )

    train_percentage = 0.9
    for source, gdf in annotations_df.groupby("source"):
        shuffled_df = gdf.sample(n=gdf.shape[0], random_state=42)
        n_train_samples = round(shuffled_df.shape[0] * train_percentage)
        shuffled_df.iloc[:n_train_samples].to_csv(
            str(Path(BASE_DATASET_DIR, f"train_{source}.tsv")), index=False, sep="\t"
        )
        shuffled_df.iloc[n_train_samples:].to_csv(
            str(Path(BASE_DATASET_DIR, f"test_{source}.tsv")), index=False, sep="\t"
        )
