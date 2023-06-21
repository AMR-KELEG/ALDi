import math
import torch
import random
import argparse
import pandas as pd
from glob import glob
from pathlib import Path

from torch import Tensor
from transformers import Trainer, TrainingArguments
from transformers.integrations import TensorBoardCallback
from transformers import AutoTokenizer, BertForSequenceClassification

random.seed(42)
torch.manual_seed(42)


def compute_evaluation_metrics(eval_prediction):
    """Compute RMSE metric as a callback during training.

    Args:
        eval_prediction: An EvalPrediction.

    Returns:
        A dictionary of evaluation metrics to report during training.
    """
    labels = eval_prediction.label_ids.reshape(-1)
    predictions = eval_prediction.predictions.reshape(-1)
    sq_error = (labels - predictions) ** 2

    return {"RMSE": math.sqrt(sq_error.sum() / sq_error.shape[0])}


def transform_input(tokenizer, filenames):
    """Tokenize the input text and return the features in the HF dict format.

    Args:
        tokenizer: The model's tokenizer.
        filenames: A list of tsv files of AOC.

    Returns:
        Features in the form of a HF dict format
    """
    dfs = [pd.read_csv(filename, sep="\t") for filename in filenames]

    df = pd.concat(dfs)
    features_dict = tokenizer(
        df["sentence"].tolist(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    # Allow the dataset not to have a label
    if "average_dialectness_level" in df.columns:
        features_dict["labels"] = Tensor(
            df["average_dialectness_level"].tolist()
        ).reshape(-1, 1)

    return features_dict


class AOCDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dataset_file_path):
        super(AOCDataset).__init__()
        dataset_filenames = [f for f in glob(dataset_file_path)]

        self.features_dict = transform_input(tokenizer, dataset_filenames)
        self.input_keys = self.features_dict.keys()

    def __len__(self):
        return self.features_dict["labels"].shape[0]

    def __getitem__(self, idx):
        return {k: self.features_dict[k][idx, :] for k in self.input_keys}


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
        "--train",
        "-d",
        required=True,
        help="The filename of the training dataset (allows for glob).",
    )
    training_subparser.add_argument(
        "--dev",
        required=False,
        help="The filename of the development dataset (allows for glob).",
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
        help="The output filename.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.mode == "train":
        model = BertForSequenceClassification.from_pretrained(
            args.model_name, num_labels=1
        )
        # TODO: Update the training arguments
        NO_STEPS = 1000
        BATCH_SIZE = 32
        training_args = TrainingArguments(
            output_dir=args.o,
            save_strategy="epoch",
            eval_steps=NO_STEPS,
            per_device_train_batch_size=32,
            evaluation_strategy="steps",
        )
        train_dataset = AOCDataset(tokenizer, args.train)

        eval_dataset_filenames = glob(args.dev)
        eval_dataset = {
            Path(filename).name.split(".")[0]: AOCDataset(tokenizer, filename)
            for filename in eval_dataset_filenames
        }
        # eval_dataset = AOCDataset(tokenizer, args.dev)
        trainer = Trainer(
            model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[TensorBoardCallback],
            compute_metrics=compute_evaluation_metrics,
        )
        trainer.train()
    else:
        # Load from checkpoint
        model = BertForSequenceClassification.from_pretrained(args.p, num_labels=1)
        test_dataset = AOCDataset(tokenizer, args.d)

        trainer = Trainer(model, args=None, train_dataset=None)
        predictions = trainer.predict(test_dataset)
        prediction_logits = predictions.predictions.reshape(-1).tolist()

        df = pd.read_csv(args.d, sep="\t")
        df["prediction"] = prediction_logits

        df.to_csv(args.o, index=False, sep="\t")


if __name__ == "__main__":
    main()
