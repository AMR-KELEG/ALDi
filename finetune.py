import evaluate
import numpy as np
from finetune_utils import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"], padding="max_length", max_length=64, truncation=True
    )


def main():
    MODEL_NAME = "UBC-NLP/MARBERT"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.cuda()

    datasets = {split: load_dataset(split) for split in ["train", "dev", "test"]}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Shuffling is needed for more stable fine-tuning?!

    tokenized_datasets = {
        split: datasets[split]
        .map(lambda sample: tokenize_function(sample, tokenizer), batched=True)
        .shuffle(seed=42)
        for split in datasets.keys()
    }

    training_args = TrainingArguments(
        output_dir="MARBERT_DI",
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=32,
        learning_rate=2e-5,
        num_train_epochs=20,
        logging_strategy="steps",
        save_strategy="steps",
        logging_steps=10,
        save_steps=10,
        load_best_model_at_end=True,
        # report_to="wandb",
    )
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"],
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()
