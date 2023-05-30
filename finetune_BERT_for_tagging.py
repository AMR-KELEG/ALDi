import torch
from torch import nn
from dataset_loaders import load_LinCE
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForTokenClassification


class LIDataset:
    def __init__(
        self,
        tokenizer,
        dataset,
        max_seq_len,
    ):
        # The dataset as a list of sentences
        # [(tokens_1, tags_1), (tokens_2, tags_2), ....]

        self.texts = [s[0] for s in dataset]
        self.tags = [s[1] for s in dataset]

        # Encode the BIL tags
        self.label_list = sorted(
            set([tag for tag_list in self.tags for tag in tag_list])
        )
        self.label_map = {label: i for i, label in enumerate(self.label_list)}

        # A wordpiece tokenizer
        self.tokenizer = tokenizer

        self.max_seq_len = max_seq_len

        # Use cross entropy ignore_index as padding label id so that only
        # real label ids contribute to the loss later.
        self.pad_token_label_id = nn.CrossEntropyLoss().ignore_index

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        # TODO: This is a redundant computation
        # Is it worth refactoring?
        words = self.texts[index]
        labels = self.tags[index]

        # Tokens and Labels of current example
        tokens, label_ids = [], []
        for word, label in zip(words, labels):
            word_tokens = self.tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                # Append the list of subtokens
                tokens.extend(word_tokens)

                # Ignore the subwords
                subwords_label = self.pad_token_label_id

                label_ids.extend(
                    # Label for first subtoken
                    [self.label_map[label]]
                    +
                    # Padding for the rest of subtokens
                    [subwords_label] * (len(word_tokens) - 1)
                )

        # Truncate the sample while reserving two tokens for [CLS] and [SEP]
        if len(tokens) > self.max_seq_len - 2:
            # TODO: Add a debugging message
            tokens = tokens[: (self.max_seq_len - 2)]
            label_ids = label_ids[: (self.max_seq_len - 2)]

        # Add the [CLS] and [SEP] tokens
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        label_ids = [self.pad_token_label_id] + label_ids + [self.pad_token_label_id]
        segment_ids = [0] * len(tokens)

        # Encode the subwords to indecies
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens
        # Only real tokens are attended to
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.max_seq_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        # Don't attend to the padding
        input_mask += [0] * padding_length
        segment_ids += [0] * padding_length
        # Don't use padding on computing loss
        label_ids += [self.pad_token_label_id] * padding_length

        assert len(input_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len
        assert len(label_ids) == self.max_seq_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(input_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(segment_ids, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


if __name__ == "__main__":
    model_name = "UBC-NLP/MARBERT"

    dataset = load_LinCE("dev")
    TAGS = ["ambiguous", "lang1", "lang2", "mixed", "ne", "other"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(TAGS)
    )

    train_dataset = LIDataset(
        tokenizer=tokenizer, dataset=load_LinCE("train"), max_seq_len=512
    )
    eval_dataset = LIDataset(
        tokenizer=tokenizer, dataset=load_LinCE("dev"), max_seq_len=512
    )

    NO_STEPS = 100
    training_args = TrainingArguments(
        output_dir="BERT_for_tagging",
        save_strategy="epoch",
        eval_steps=NO_STEPS,
        evaluation_strategy="steps",
    )

    # Make sure it is using the right optimization function
    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
