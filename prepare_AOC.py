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
    """Augment the columns of the dataframe.

    Args:
        df: A dataframe of AOC samples.

    Returns:
        An augmented version of the dataframe.
    """
    df["#_tokens"] = df["Sentence"].apply(lambda s: len(str(s).split()))

    sentences_count = Counter(df["Sentence"])
    df["#_annotations_per_sentence"] = df["Sentence"].apply(
        lambda s: sentences_count[s]
    )

    #  Document ID can be repeated in multiple sources but referring to different documents
    df["document"] = df.apply(lambda row: f'{row["source"]}_{row["document"]}', axis=1)

    # Generate an ID for each sentence to refer to it later
    sentences_id = {s: i for i, (s, c) in enumerate(sentences_count.most_common())}
    df["SID"] = df.apply(
        lambda row: f'{sentences_id[row["Sentence"]]}_{row["document"]}', axis=1
    )

    sentences_per_document_count = Counter(df["SID"])
    df["#_annotations"] = df["SID"].apply(lambda sid: sentences_per_document_count[sid])

    # Find the set of sentences marked at least one as junk or without a label
    junk_sentences = set(
        df.loc[df["DLevel"].isin(["junk", "prompt"]), "Sentence"].tolist()
    )
    df["is_junk"] = df["Sentence"].apply(lambda s: s in junk_sentences)

    # Map the categorical dialectness levels to numeric values
    dialectness_level_map = {"most": 1, "mixed": 2 / 3, "little": 1 / 3, "msa": 0}
    df["dialectness_level"] = df["DLevel"].apply(
        lambda l: dialectness_level_map.get(l, None)
    )

    # Categorize the sentences based on the sentence length
    df["sentence_length"] = df["#_tokens"].apply(
        lambda no_tokens: "short"
        if no_tokens < 5
        else "long"
        if no_tokens > 27
        else "medium"
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
        key=lambda row: row["SID"],
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

        annotations.append(
            {
                "sentence": sentence,
                "sentence_length": group_items[0]["sentence_length"],
                "dialect_level": dialect_level,
                "dialect": dialect_class,
                "dialectness_level": dialectness_level,
                "average_dialectness_level": sum(dialectness_level)
                / len(dialectness_level),
                "same_polarity": all([s == 0 for s in dialectness_level])
                or all([s != 0 for s in dialectness_level]),
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

    # Filter samples keeping only sentences satisfying the following criteria:
    # 1) Sentences that are not mark as junk by any of the 3 annotators
    # 2) Sentences that are not extracted from articles' bodies
    single_sentence_comment_df = (
        df[(~df["is_junk"]) & (~df["is_control_sentence"])]
        .copy()
        .reset_index(drop=True)
    )

    sentences = set(single_sentence_comment_df["Sentence"].tolist())
    discarded_df = df[
        df["Sentence"].progress_apply(lambda s: s not in sentences)
    ].copy()

    discarded_df["is_not_a_sentence"] = discarded_df["sentence_length"] != "medium"
    discarded_df["has_annotations_issue"] = discarded_df["#_annotations"] != 3
    discarded_df.to_csv(
        str(Path(BASE_DATASET_DIR, "AOC_discarded.tsv")), index=False, sep="\t"
    )

    annotations_df = group_annotations_by_sentence_id(single_sentence_comment_df)
    annotations_df.to_csv(
        str(Path(BASE_DATASET_DIR, "AOC_aggregated.tsv")), index=False, sep="\t"
    )

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

    # (5) Make sure that the comments to the same article are within the same split
    for source, gdf in annotations_df.groupby("source"):
        gdf.sort_values(
            by=["comments_per_document", "document"], inplace=True, ascending=False
        )

        train_df = generate_sliced_df(gdf, train_size)
        train_df.to_csv(
            str(Path(BASE_DATASET_DIR, f"train_{source}.tsv")), index=False, sep="\t"
        )

        dev_df = generate_sliced_df(gdf, train_size + dev_size).iloc[
            train_df.shape[0] :
        ]
        dev_df.to_csv(
            str(Path(BASE_DATASET_DIR, f"dev_{source}.tsv")), index=False, sep="\t"
        )

        test_df = gdf.iloc[train_df.shape[0] + dev_df.shape[0] :]
        test_df.to_csv(
            str(Path(BASE_DATASET_DIR, f"test_{source}.tsv")), index=False, sep="\t"
        )


if __name__ == "__main__":
    main()
