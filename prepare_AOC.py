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

import argparse

BASE_DATASET_DIR = "data/AOC"
RAW_DATASET_FILE = str(Path(BASE_DATASET_DIR, "dialectid_data.tsv"))
DATASET_URL = (
    "https://raw.githubusercontent.com/sjeblee/AOC/master/"
    + "stuff-from-omar/annotated-aoc-data/dialectid_data.tsv"
)


def download_AOC():
    # Create the dataset directory
    os.makedirs(BASE_DATASET_DIR, exist_ok=True)

    # Retrieve the dataset
    urlretrieve(DATASET_URL, RAW_DATASET_FILE)


def explode_AOC():
    """Explode HIT annotation rows into single annotation ones."""

    # Loading the dataset using pandas causes issues
    # TODO: Is that a reason of " within some lines?
    with open(RAW_DATASET_FILE, "r") as f:
        lines = [l.strip() for l in f]

    fields = [l.split("\t") for l in tqdm(lines)]
    assert len(set([len(f) == 100 for f in fields]))

    df = pd.DataFrame(data=fields[1:], columns=fields[0])

    # Columns repeated 12 times per HIT
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
    """Augment the columns of the dataframe.

    Args:
        df: A dataframe of AOC samples.

    Returns:
        An augmented version of the dataframe.
    """
    df["#_tokens"] = df["Sentence"].apply(lambda s: len(str(s).split()))
    df["length"] = df["#_tokens"].apply(
        lambda n: "short" if n < 5 else "long" if n > 27 else "medium"
    )

    # Generate an ID for each sentence to refer to it later
    sentences_count = Counter(df["Sentence"])
    sentences_id = {s: i for i, (s, c) in enumerate(sentences_count.most_common())}
    df["SENTENCE_ID"] = df["Sentence"].apply(lambda s: sentences_id[s])
    df["document"] = df["document"].apply(
        lambda d: int(d) if str(d) != "nan" else "nan"
    )
    df["COMMENT_ID"] = df.apply(
        lambda row: f"{row['source']}_{row['document']}_{row['SENTENCE_ID']}", axis=1
    )

    # Map the categorical dialectness levels to numeric values
    dialectness_level_map = {"most": 1, "mixed": 2 / 3, "little": 1 / 3, "msa": 0}
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
    # Form the annotations df - grouped by the generated comment id
    annotations = []
    df.sort_values(by="COMMENT_ID", inplace=True)
    for comment_id, group in groupby(
        [row for i, row in df.iterrows()],
        key=lambda row: row["COMMENT_ID"],
    ):
        group_items = list(group)
        group_items = sorted(group_items, key=lambda d: d["AID"])

        assert len(set([item["Sentence"] for item in group_items])) == 1

        sentence = group_items[0]["Sentence"]
        comment_id = group_items[0]["COMMENT_ID"]

        dialect_level = [
            None
            if str(group_item["DLevel"]) in ["nan", "prompt"]
            else str(group_item["DLevel"])
            for group_item in group_items
        ]

        # Ignore annotations with missing dialect level
        group_items_with_level = [
            item for item, level in zip(group_items, dialect_level) if level
        ]

        dialect_level = [
            None
            if str(group_item["DLevel"]) in ["nan", "prompt"]
            else str(group_item["DLevel"])
            for group_item in group_items_with_level
        ]
        dialect_class = [
            group_item["DClass"]
            if str(group_item["DClass"]) not in ["nan", "prompt"]
            else group_item["DLevel"]
            if str(group_item["DLevel"]) in ["msa", "junk"]
            else None
            for group_item in group_items_with_level
        ]

        native_dialect = [
            str(group_item["native_dialect"]) for group_item in group_items_with_level
        ]
        native_arabic_speaker = [
            str(group_item["native_arabic"]) for group_item in group_items_with_level
        ]
        annotator_residence = [
            group_item["country_live"] for group_item in group_items_with_level
        ]
        annotator_city_from_IP = [
            group_item["WorkerCity"] for group_item in group_items_with_level
        ]
        annotator_country_from_IP = [
            group_item["WorkerCountry"] for group_item in group_items_with_level
        ]

        dialectness_level = [
            group_item["dialectness_level"] for group_item in group_items_with_level
        ]

        has_junk_annotation = "junk" in dialect_level

        if not dialectness_level:
            continue

        annotations.append(
            {
                "annotator_city_from_IP": annotator_city_from_IP,
                "annotator_country_from_IP": annotator_country_from_IP,
                "annotator_dialect": native_dialect,
                "annotator_id": [
                    group_item["AID"] for group_item in group_items_with_level
                ],
                "annotator_residence": annotator_residence,
                "annotator_native_arabic_speaker": native_arabic_speaker,
                "average_dialectness_level": sum(dialectness_level)
                / len(dialectness_level),
                "comment_id": comment_id,
                "dialect_level": dialect_level,
                "dialect": dialect_class,
                "dialectness_level": dialectness_level,
                "document": group_items_with_level[0]["document"],
                "has_junk_annotation": has_junk_annotation,
                "length": group_items_with_level[0]["length"],
                "number_annotations": len(dialect_level),
                "same_label": len(set(dialectness_level)) == 1,
                "same_polarity": all([s == 0 for s in dialectness_level])
                or all([s != 0 for s in dialectness_level]),
                "sentence": sentence,
                "source": group_items_with_level[0]["source"],
            }
        )

    annotations_df = pd.DataFrame(annotations)
    return annotations_df


def generate_sliced_df(df, percentage_within):
    """Find the part of the dataset covering at least "percentage_within"% of the dataset.

    Args:
        df: The dataframe to slice sorted according to the "document" column.
        percentage_within: The percentage of the dataframe to slice.

    Returns:
        A dataframe of the required percentage size.
    """

    n_required = int(percentage_within * df.shape[0])
    doc_comments_counts = df["document"].value_counts().tolist()
    doc_cum_counts = doc_comments_counts

    for i in range(1, len(doc_cum_counts)):
        doc_cum_counts[i] += doc_cum_counts[i - 1]
        if doc_cum_counts[i] >= n_required:
            return df.iloc[: doc_cum_counts[i]]

    return df


def main():
    parser = argparse.ArgumentParser("Form AOC data splits.")
    parser.add_argument(
        "--train_size",
        default=0.8,
        help="Percentage of the dataset to be used for the training split.",
    )
    parser.add_argument(
        "--dev_size",
        default=0.1,
        help="Percentage of the dataset to be used for the development split.",
    )

    args = parser.parse_args()
    train_size = args.train_size
    dev_size = args.dev_size

    download_AOC()

    df = explode_AOC()
    df.to_csv(str(Path(BASE_DATASET_DIR, "AOC_exploded.tsv")), index=False, sep="\t")

    df = augment_AOC_cols(df)

    annotations_df = group_annotations_by_sentence_id(df)
    annotations_df.to_csv(
        str(Path(BASE_DATASET_DIR, "AOC_aggregated.tsv")), index=False, sep="\t"
    )

    # Filter out samples having a junk annotation
    annotations_df = annotations_df[~annotations_df["has_junk_annotation"]]

    # TODO: Filter out samples from articles themselves?

    # Find the number of comments for each document in the corpus
    comments_per_document = (
        annotations_df["document"]
        .value_counts()
        .reset_index()
        .to_dict(orient="records")
    )
    comments_per_document = {d["index"]: d["document"] for d in comments_per_document}
    annotations_df["comments_per_document"] = annotations_df["document"].apply(
        lambda d: comments_per_document[d]
    )

    annotations_df.sort_values(
        by=["comments_per_document", "document"], inplace=True, ascending=False
    )

    train_df = generate_sliced_df(annotations_df, train_size)
    train_df.to_csv(str(Path(BASE_DATASET_DIR, f"train.tsv")), index=False, sep="\t")

    dev_df = generate_sliced_df(annotations_df, train_size + dev_size).iloc[
        train_df.shape[0] :
    ]
    dev_df.to_csv(str(Path(BASE_DATASET_DIR, f"dev.tsv")), index=False, sep="\t")

    test_df = annotations_df.iloc[train_df.shape[0] + dev_df.shape[0] :]
    test_df.to_csv(str(Path(BASE_DATASET_DIR, f"test.tsv")), index=False, sep="\t")


if __name__ == "__main__":
    main()
