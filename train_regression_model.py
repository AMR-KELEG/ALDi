from torch import Tensor
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, BertForSequenceClassification

import argparse

import torch
import random
import pandas as pd

random.seed(42)
torch.manual_seed(42)


def transform_input(tokenizer, filename):
    df = pd.read_csv(filename, sep="\t")

    # TODO: Adapt this mapping into the dataset creation script
    # Map "+2" to "0"
    df["mapped_dialectness_level"] = df["dialectness_level"].apply(
        lambda l: [0 if float(v) == 2 else -float(v) for v in l[1:-1].split(",")]
    )

    # Map scores from [0, 3] to [0, 1]
    df["mapped_average_dialectness_level"] = df["mapped_dialectness_level"].apply(
        lambda l: (sum(l) / len(l)) / 3
    )

    features_dict = tokenizer(
        df["sentence"].tolist(), return_tensors="pt", padding=True
    )
    features_dict["labels"] = Tensor(df["mapped_average_dialectness_level"].tolist())
    return features_dict


class AOCDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, filename):
        super(AOCDataset).__init__()
        self.features_dict = transform_input(tokenizer, filename)
        self.input_keys = self.features_dict.keys()

    def __len__(self):
        return self.features_dict["labels"].shape[0]

    def __getitem__(self, idx):
        return {
            k: self.features_dict[k][idx, :]
            if len(self.features_dict[k].shape) > 1
            else self.features_dict[k][idx]
            for k in self.input_keys
        }


def main():
    parser = argparse.ArgumentParser(
        "Train a regression model for determining the level of dialcetness."
    )

    subparsers = parser.add_subparsers()
    training_subparser = subparsers.add_parser(
        "train",
        help="Train the regression model.",
    )
    training_subparser.set_defaults(mode="train")
    training_subparser.add_argument(
        "-d",
        required=True,
        help="The path of the training dataset.",
    )
    training_subparser.add_argument(
        "-model_name",
        "-m",
        default="UBC-NLP/MARBERT",
        help="The model name.",
    )
    training_subparser.add_argument(
        "-o",
        required=True,
        help="The output directory.",
    )

    prediction_subparser = subparsers.add_parser(
        "predict",
        help="Generate predictions using the regression model.",
    )
    prediction_subparser.set_defaults(mode="predict")
    prediction_subparser.add_argument(
        "-d",
        required=True,
        help="The path of the dataset.",
    )
    prediction_subparser.add_argument(
        "-model_name",
        "-m",
        default="UBC-NLP/MARBERT",
        help="The model name.",
    )
    prediction_subparser.add_argument(
        "-p",
        required=True,
        help="The trained model path.",
    )
    prediction_subparser.add_argument(
        "-o",
        required=True,
        help="The output directory.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.mode == "train":
        model = BertForSequenceClassification.from_pretrained(
            args.model_name, num_labels=1
        )
        # TODO: Update the training arguments
        training_args = TrainingArguments(output_dir=args.o, save_strategy="epoch")
        # "data/AOC/train_youm7_c.tsv"
        train_dataset = AOCDataset(tokenizer, args.d)
        trainer = Trainer(model, args=training_args, train_dataset=train_dataset)
        trainer.train()
    else:
        # Load from checkpoint
        model = BertForSequenceClassification.from_pretrained(args.p, num_labels=1)
        # "data/AOC/test_youm7_c.tsv"
        test_dataset = AOCDataset(tokenizer, args.d)

        trainer = Trainer(model, args=None, train_dataset=test_dataset)
        predictions = trainer.predict(test_dataset)
        prediction_logits = predictions.predictions.reshape(-1).tolist()

        df = pd.read_csv(args.d, sep="\t")
        df["prediction"] = prediction_logits

        # TODO: Adapt this mapping into the dataset creation script
        # Map "+2" to "0"
        df["mapped_dialectness_level"] = df["dialectness_level"].apply(
            lambda l: [0 if float(v) == 2 else -float(v) for v in l[1:-1].split(",")]
        )

        # Map scores from [0, 3] to [0, 1]
        df["mapped_average_dialectness_level"] = df["mapped_dialectness_level"].apply(
            lambda l: (sum(l) / len(l)) / 3
        )

        df.to_csv("TODO.tsv", index=False, sep="\t")


if __name__ == "__main__":
    main()
