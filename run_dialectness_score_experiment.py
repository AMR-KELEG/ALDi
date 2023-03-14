import os
import argparse
from dataset_loaders import load_AOC, load_BIBLE, load_DIAL2MSA
from metrics import BackTranslationMetric, LexiconOverlapMetric
from pathlib import Path
from tqdm import tqdm

tqdm.pandas()

DATASET_LOADING_FUNCTION = {
    "AOC": load_AOC,
    "BIBLE": load_BIBLE,
    "DIAL2MSA": load_DIAL2MSA,
}
DIALECTNESS_METRIC = {
    "backtranslation": BackTranslationMetric,
    "lexicon": LexiconOverlapMetric,
}


def main():
    parser = argparse.ArgumentParser(
        "Compute the dialectness score for a specific dataset."
    )
    parser.add_argument(
        "-dataset",
        "-d",
        choices=[
            "AOC",
            "BIBLE",
            "DIAL2MSA",
        ],
        required=True,
        help="The dataset to compute the scores for.",
    )
    parser.add_argument(
        "-metric",
        "-m",
        choices=["backtranslation", "lexicon"],
        required=True,
        help="The dialectness level metric.",
    )
    # TODO: Use subparsers
    parser.add_argument(
        "-lexicon_source",
        help="Source that was used to form the MSA lexicon.",
        choices=["UN", "opensubtitle"],
    )
    parser.add_argument(
        "-dialect_or_source",
        required=True,
        help="The dialect/source of the dataset to load.",
    )
    parser.add_argument("-split", required=True, help="The dataset split to load.")

    parser.add_argument(
        "-results_dir", required=True, help="Directory to save the results to."
    )
    parser.add_argument("-o", required=True, help="Output filename.")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    if args.metric == "lexicon":
        metric = DIALECTNESS_METRIC[args.metric](lexicon_source=args.lexicon_source)
    else:
        metric = DIALECTNESS_METRIC[args.metric]()

    if args.dataset == "AOC":
        dataset = DATASET_LOADING_FUNCTION[args.dataset](
            split=args.split, source=args.dialect_or_source
        )
    else:
        dataset = DATASET_LOADING_FUNCTION[args.dataset](
            split=args.split, dialect=args.dialect_or_source
        )

    # TODO: Change the name of the column in the original tsv file
    dataset.rename(columns={"sentence": "DA_text"}, inplace=True)

    dataset["DA_score"] = dataset["DA_text"].progress_apply(
        lambda s: metric.compute_dialectness_score(s)
    )

    if args.dataset != "AOC":
        dataset["MSA_score"] = dataset["MSA_text"].progress_apply(
            lambda s: metric.compute_dialectness_score(s)
        )
        dataset["delta_score"] = dataset["DA_score"] - dataset["MSA_score"]
    dataset.to_csv(str(Path(args.results_dir, args.o)), sep="\t", index=False)


if __name__ == "__main__":
    main()
