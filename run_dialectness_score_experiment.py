import os
import argparse
from dataset_loaders import load_AOC, load_BIBLE, load_DIAL2MSA, load_contrastive_pairs
from metrics import (
    BackTranslationMetric,
    LexiconOverlapMetric,
    RegressionBERTMetric,
    LIBERTMetric,
)
from pathlib import Path
from tqdm import tqdm

tqdm.pandas()

DATASET_LOADING_FUNCTION = {
    "AOC": load_AOC,
    "BIBLE": load_BIBLE,
    "DIAL2MSA": load_DIAL2MSA,
    "CONTRAST": load_contrastive_pairs,
}
DIALECTNESS_METRIC = {
    "backtranslation": BackTranslationMetric,
    "lexicon": LexiconOverlapMetric,
    "regression": RegressionBERTMetric,
    "tagging": LIBERTMetric,
}


def main():
    parser = argparse.ArgumentParser(
        "Compute the dialectness score for a specific dataset."
    )
    parser.add_argument(
        "-dataset",
        "-d",
        choices=sorted(DATASET_LOADING_FUNCTION.keys()),
        required=True,
        help="The dataset to compute the scores for.",
    )
    parser.add_argument(
        "-metric",
        "-m",
        choices=sorted(DIALECTNESS_METRIC.keys()),
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
        "-use_medium_length",
        help="Filter out short and long samples from AOC.",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-model_name",
        help="The name of the pretrained BERT model.",
        default="UBC-NLP/MARBERT",
        required=False,
    )
    parser.add_argument(
        "-model_path",
        help="The path to the fine-tuned BERT model.",
        required=False,
    )
    parser.add_argument(
        "-dialect_or_source",
        default=None,
        help="The dialect/source of the dataset to load.",
    )
    parser.add_argument("-split", help="The dataset split to load.")

    parser.add_argument(
        "-results_dir", required=True, help="Directory to save the results to."
    )
    parser.add_argument("-o", required=True, help="Output filename.")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    if args.metric == "lexicon":
        metric = DIALECTNESS_METRIC[args.metric](lexicon_source=args.lexicon_source)
    elif args.metric == "backtranslation":
        metric = DIALECTNESS_METRIC[args.metric]()
    else:
        metric = DIALECTNESS_METRIC[args.metric](
            model_path=args.model_path, model_name=args.model_name
        )

    if args.dataset == "AOC":
        dataset = DATASET_LOADING_FUNCTION[args.dataset](
            split=args.split, source=args.dialect_or_source
        )
    elif args.dataset == "CONTRAST":
        dataset = DATASET_LOADING_FUNCTION[args.dataset]()
    else:
        dataset = DATASET_LOADING_FUNCTION[args.dataset](dialect=args.dialect_or_source)

    # TODO: Change the name of the column in the original tsv file
    dataset.rename(columns={"sentence": "DA_text"}, inplace=True)

    # Filter out short and long samples from AOC
    if args.dataset == "AOC" and args.use_medium_length:
        dataset = dataset[dataset["sentence_length"] == "medium"].copy()

    if "MSA_text" in dataset.columns:
        dataset["MSA_score"] = dataset["MSA_text"].progress_apply(
            lambda s: metric.compute_dialectness_score(s)
        )

    dataset["DA_score"] = dataset["DA_text"].progress_apply(
        lambda s: metric.compute_dialectness_score(s)
    )

    if "MSA_text" in dataset.columns:
        dataset["delta_score"] = dataset["DA_score"] - dataset["MSA_score"]
    dataset.to_csv(str(Path(args.results_dir, args.o)), sep="\t", index=False)


if __name__ == "__main__":
    main()
