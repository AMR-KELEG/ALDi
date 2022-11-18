import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import numpy as np
from pathlib import Path
from datasets import Dataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from scipy.special import softmax


def load_dataset(dialectness_level, DATASET):
    BASEDIR = f"data/{DATASET}_mapped/"
    DA_file = str(Path(BASEDIR, f"{dialectness_level}_DA.txt"))
    MSA_file = str(Path(BASEDIR, f"{dialectness_level}_MSA.txt"))

    with open(DA_file, "r") as f:
        DA_samples = [{"text": l.strip(), "label": 1} for l in f]

    with open(MSA_file, "r") as f:
        MSA_samples = [{"text": l.strip(), "label": 0} for l in f]

    return Dataset.from_list(DA_samples + MSA_samples)


def tokenize_function(examples, tokenizer):
    # return tokenizer(examples["text"], padding="max_length", truncation=True)
    return tokenizer(
        examples["text"], padding="max_length", max_length=64, truncation=True
    )


def predict(sample):
    MODEL_NAME = "UBC-NLP/MARBERT"
    model = AutoModelForSequenceClassification.from_pretrained(
        "MARBERT_DI/checkpoint-240/"
    )
    model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_samples = tokenize_function(
        Dataset.from_list([{"text": sample}]), tokenizer
    )

    predictions = model(
        input_ids=torch.tensor(tokenized_samples["input_ids"]).cuda(),
        attention_mask=torch.tensor(tokenized_samples["attention_mask"]).cuda(),
        token_type_ids=torch.tensor(tokenized_samples["token_type_ids"]).cuda(),
    )

    return softmax(predictions.logits.cpu().tolist())


def main(dialectness_level, DATASET):
    MODEL_NAME = "UBC-NLP/MARBERT"
    model = AutoModelForSequenceClassification.from_pretrained(
        "MARBERT_DI/checkpoint-240/"
    )
    model.cuda()

    eval_dataset = load_dataset(dialectness_level, DATASET)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_samples = eval_dataset.map(
        lambda sample: tokenize_function(sample, tokenizer), batched=True
    )

    predicted_labels = []
    prediction_confidence_scores = []
    BATCH_SIZE = 32
    for start_index in tqdm(range(0, len(tokenized_samples), BATCH_SIZE)):
        batch_samples = tokenized_samples[start_index : start_index + BATCH_SIZE]
        predictions = model(
            input_ids=torch.tensor(batch_samples["input_ids"]).cuda(),
            attention_mask=torch.tensor(batch_samples["attention_mask"]).cuda(),
            token_type_ids=torch.tensor(batch_samples["token_type_ids"]).cuda(),
        )
        predicted_labels += predictions.logits.cpu().argmax(dim=1).tolist()
        prediction_confidence_scores += (
            softmax(predictions.logits.cpu().detach().numpy(), axis=1)
            .max(axis=1)
            .tolist()
        )

    eval_dataset = eval_dataset.add_column("prediction", predicted_labels)
    eval_dataset = eval_dataset.add_column(
        "prediction_score", prediction_confidence_scores
    )
    return eval_dataset


if __name__ == "__main__":
    DATASET = "MADAR"
    # DATASET = "DIA2MSA"
    for dialectness_level in ["little", "most"]:
        dataset = main(dialectness_level, DATASET)
        df = pd.DataFrame(
            {
                "text": dataset["text"],
                "label": dataset["label"],
                "prediction": dataset["prediction"],
                "prediction_score": dataset["prediction_score"],
            }
        )
        df.to_csv(f"data/output_eval_{DATASET}_{dialectness_level}.csv", index=False)
