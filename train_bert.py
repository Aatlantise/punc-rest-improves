import re
import os
import torch
import numpy as np
from argparse import ArgumentParser
from datasets import load_from_disk
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer, 
    BertModel, 
    BertPreTrainedModel, 
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback
)

# --- Configuration ---
# MODEL_CHECKPOINT = "bert-base-uncased"
MODEL_CHECKPOINT = "./pr-bert-financial-news"
DATASET_NAME = "data/finance-news"
MAX_SEQ_LENGTH = 512
OVERLAP = 100
BATCH_SIZE = 128
LEARNING_RATE = 3e-5
EPOCHS = 30
# OUTPUT_DIR = "./pr-bert-financial-news"
# LOGGING_DIR = "./logs"
OUTPUT_DIR = "./pr-bert-financial-news-continued"
LOGGING_DIR = "./logs/bert-continued"
EARLY_STOPPING_PATIENCE = 3

# labels
PUNCTUATION = ['.', ',', '!', '?', ';', ':', '"', "'"]
CAPITALIZATION = ["LOWER", "UPPER", "CAPITALIZED", "OTHER"]
PUNCTUATION_REGEX = re.compile(r'[.,!?;:"\']')


# --- Preprocessing Helpers ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())

def get_capitalization_label(token):
    letters = "".join(char for char in token if char.isalpha())
    if not letters:
        return 0
    if letters.islower():
        return 0
    if letters.isupper():
        return 1
    if letters[0].isupper() and letters[1:].islower():
        return 2
    return 3

def create_word_labels(text):
    words, punct_labels, cap_labels = [], [], []
    for token in clean_text(text).split():
        word = PUNCTUATION_REGEX.sub("", token).lower()
        if not word:
            continue
        words.append(word)
        punct_labels.append([int(punct in token) for punct in PUNCTUATION])
        cap_labels.append(get_capitalization_label(token))
    return words, punct_labels, cap_labels

def sliding_window_preprocess(examples, tokenizer):
    model_inputs = {"input_ids": [], "attention_mask": [], "token_type_ids": [],
                    "punct_labels": [], "cap_labels": []}
    for text in examples["text"]:
        words, word_punct_labels, word_cap_labels = create_word_labels(text)
        if len(words) < 5:
            continue
        encoded = tokenizer(
            words,
            is_split_into_words=True,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            stride=OVERLAP,
            return_overflowing_tokens=True
        )
        # how many windows are made using the given overlap
        for window_index in range(len(encoded["input_ids"])):
            # if single word has multiple tokens, it has same word id
            word_ids = encoded.word_ids(batch_index=window_index)
            punct_labels, cap_labels = [], []
            previous_word_id = None
            for word_id in word_ids:
                if word_id is None or word_id == previous_word_id:
                    punct_labels.append([-100] * len(PUNCTUATION))
                    cap_labels.append(-100)
                else:
                    punct_labels.append(word_punct_labels[word_id])
                    cap_labels.append(word_cap_labels[word_id])
                previous_word_id = word_id
            model_inputs["input_ids"].append(encoded["input_ids"][window_index])
            model_inputs["attention_mask"].append(encoded["attention_mask"][window_index])
            model_inputs["token_type_ids"].append(encoded["token_type_ids"][window_index])
            model_inputs["punct_labels"].append(punct_labels)
            model_inputs["cap_labels"].append(cap_labels)
    return model_inputs

class DualHeadDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # features is a list each entry is {"input ids": , "attention_mask":..., "punct_labels":, ...}
        punct_labels = [feature["punct_labels"] for feature in features]
        cap_labels = [feature["cap_labels"] for feature in features]
        model_features = [{key: value for key, value in feature.items()
                           if key not in {"punct_labels", "cap_labels"}} for feature in features]
        batch = self.tokenizer.pad(model_features, 
                                   padding=True, 
                                   pad_to_multiple_of=8, 
                                   return_tensors="pt")
        sequence_length = batch["input_ids"].shape[1]
        punct_padding = [-100] * len(PUNCTUATION)
        if self.tokenizer.padding_side == "right":
            punct_labels = [labels + [punct_padding] * (sequence_length - len(labels)) for labels in punct_labels]
            cap_labels = [labels + [-100] * (sequence_length - len(labels)) for labels in cap_labels]
        else:
            punct_labels = [[punct_padding] * (sequence_length - len(labels)) + labels for labels in punct_labels]
            cap_labels = [[-100] * (sequence_length - len(labels)) + labels for labels in cap_labels]
        batch["punct_labels"] = torch.tensor(punct_labels, dtype=torch.float)
        batch["cap_labels"] = torch.tensor(cap_labels, dtype=torch.long)
        return batch

# --- Model ---

class BertForPunctuationCapitalization(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.num_punctuation_labels = len(PUNCTUATION)
        config.num_capitalization_labels = len(CAPITALIZATION)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.punct_classifier = torch.nn.Linear(config.hidden_size, len(PUNCTUATION))
        self.cap_classifier = torch.nn.Linear(config.hidden_size, len(CAPITALIZATION))
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                punct_labels=None, cap_labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, **kwargs)
        sequence_output = self.dropout(outputs.last_hidden_state)
        punct_logits = self.punct_classifier(sequence_output)
        cap_logits = self.cap_classifier(sequence_output)
        loss = None
        if punct_labels is not None and cap_labels is not None:
            punct_mask = punct_labels[:, :, 0] != -100
            cap_mask = cap_labels != -100
            punct_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                punct_logits[punct_mask], punct_labels[punct_mask]
            )
            cap_loss = torch.nn.functional.cross_entropy(cap_logits[cap_mask], cap_labels[cap_mask])
            loss = punct_loss + cap_loss
        return {
                    "loss": loss,
                    "punct_logits": punct_logits,
                    "cap_logits": cap_logits
                }
# --- Evaluation Helpers ---
def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    punct_logits, cap_logits = predictions
    punct_labels, cap_labels = labels
    punct_mask = punct_labels[:, :, 0] != -100
    punct_true = punct_labels[punct_mask].astype(int)
    punct_pred = (punct_logits[punct_mask] >= 0).astype(int)
    cap_mask = cap_labels != -100
    cap_true = cap_labels[cap_mask].astype(int)
    cap_pred = np.argmax(cap_logits[cap_mask], axis=-1)
    results = {
        "punct_micro_f1": f1_score(punct_true, punct_pred, average="micro", zero_division=0),
        "punct_macro_f1": f1_score(punct_true, punct_pred, average="macro", zero_division=0),
        "cap_micro_f1": f1_score(cap_true, cap_pred, average="micro",
                                  labels=range(len(CAPITALIZATION)), zero_division=0),
        "cap_macro_f1": f1_score(cap_true, cap_pred, average="macro",
                                  labels=range(len(CAPITALIZATION)), zero_division=0)
    }
    results["combined_macro_f1"] = (results["punct_macro_f1"] + results["cap_macro_f1"]) / 2
    return results

def main(debug=False):
    # 1. Setup
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
    model = BertForPunctuationCapitalization.from_pretrained(MODEL_CHECKPOINT)

    # 2. Split articles before making windows so overlapping windows cannot leak
    print("Loading and processing data...")
    dataset = load_from_disk(DATASET_NAME)
    if debug:
        dataset = dataset.select(range(min(1000, len(dataset))))
    split = dataset.train_test_split(test_size=0.1, seed=42)
    processed_train = split["train"].map(
        lambda x: sliding_window_preprocess(x, tokenizer),
        batched=True,
        batch_size=100,
        remove_columns=split["train"].column_names,
        num_proc=4
    )
    processed_eval = split["test"].map(
        lambda x: sliding_window_preprocess(x, tokenizer),
        batched=True,
        batch_size=100,
        remove_columns=split["test"].column_names,
        num_proc=4
    )
    epochs = 3 if debug else EPOCHS

    # 3. Training Args
    use_bf16 = torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=500,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_checkpointing=True,
        weight_decay=0.01,
        num_train_epochs=epochs,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        report_to="tensorboard",
        logging_dir=LOGGING_DIR,
        load_best_model_at_end=True,
        metric_for_best_model="combined_macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        label_names=["punct_labels", "cap_labels"]
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=processed_train,
        eval_dataset=processed_eval,
        data_collator=DualHeadDataCollator(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
    )

    # 5. Train
    print("Starting training...")
    trainer.train()
    print("Saving best model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    arguments = parser.parse_args()
    main(debug=arguments.debug)