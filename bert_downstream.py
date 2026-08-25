import argparse
import csv
import gc
import os
import random
import tempfile
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

DEFAULT_MODEL = "./pr-bert-financial-news-continued"
SEEDS = (1, 2, 3)
TASKS = ("pos", "ner", "genia", "ontonotes5")

ONTONOTES_LABEL2ID = {
    "O": 0,
    "B-CARDINAL": 1,
    "B-DATE": 2,
    "I-DATE": 3,
    "B-PERSON": 4,
    "I-PERSON": 5,
    "B-NORP": 6,
    "B-GPE": 7,
    "I-GPE": 8,
    "B-LAW": 9,
    "I-LAW": 10,
    "B-ORG": 11,
    "I-ORG": 12,
    "B-PERCENT": 13,
    "I-PERCENT": 14,
    "B-ORDINAL": 15,
    "B-MONEY": 16,
    "I-MONEY": 17,
    "B-WORK_OF_ART": 18,
    "I-WORK_OF_ART": 19,
    "B-FAC": 20,
    "B-TIME": 21,
    "I-CARDINAL": 22,
    "B-LOC": 23,
    "B-QUANTITY": 24,
    "I-QUANTITY": 25,
    "I-NORP": 26,
    "I-LOC": 27,
    "B-PRODUCT": 28,
    "I-TIME": 29,
    "B-EVENT": 30,
    "I-EVENT": 31,
    "I-FAC": 32,
    "B-LANGUAGE": 33,
    "I-PRODUCT": 34,
    "I-ORDINAL": 35,
    "I-LANGUAGE": 36,
}
ONTONOTES_ID2LABEL = {i: label for label, i in ONTONOTES_LABEL2ID.items()}


def seed_everything(seed: int) -> None:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mixed_precision_kwargs() -> Dict[str, bool]:
    use_bf16 = (
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    return {
        "bf16": bool(use_bf16),
        "fp16": bool(torch.cuda.is_available() and not use_bf16),
    }


def training_args(
    output_dir: str,
    seed: int,
    epochs: int,
    batch: int,
    learning_rate: float,
) -> TrainingArguments:
    """TrainingArguments that never save a checkpoint."""
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to="none",
        seed=seed,
        data_seed=seed,
        **mixed_precision_kwargs(),
    )


def prf1(num_correct: int, num_attempted: int, num_gold: int) -> Tuple[float, float, float]:
    precision = num_correct / num_attempted if num_attempted else 0.0
    recall = num_correct / num_gold if num_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score_pos(outputs: Iterable[List[str]], targets: Iterable[List[str]]) -> Tuple[float, float, float]:
    num_correct = 0
    num_attempted = 0
    num_gold = 0
    for pred_tags, gold_tags in zip(outputs, targets):
        num_attempted += len(pred_tags)
        num_gold += len(gold_tags)
        num_correct += sum(p == g for p, g in zip(pred_tags, gold_tags))
    return prf1(num_correct, num_attempted, num_gold)


def ner_compute_metrics(label_list: List[str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        true_predictions = []
        true_labels = []
        for pred_seq, label_seq in zip(preds, labels):
            sent_preds = []
            sent_labels = []
            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id == -100:
                    continue
                sent_preds.append(label_list[int(pred_id)])
                sent_labels.append(label_list[int(label_id)])
            true_predictions.append(sent_preds)
            true_labels.append(sent_labels)

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    return compute_metrics


def preprocess_word_labels(examples, tokenizer, label_column: str, max_length: int = 256):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=max_length,
    )

    aligned_labels = []
    for i, word_labels in enumerate(examples[label_column]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(word_labels[word_id])
            else:
                label_ids.append(-100)
            previous_word_id = word_id
        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


def evaluate_pos(model, dataset, tokenizer, label_names: List[str]) -> Tuple[float, float, float]:
    model.eval()
    device = next(model.parameters()).device
    outputs, targets = [], []

    for example in dataset:
        inputs = tokenizer(
            example["tokens"],
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(device)

        with torch.no_grad():
            preds = model(**inputs).logits.argmax(dim=-1)[0].cpu().tolist()

        word_ids = inputs.word_ids()
        pred_tags, gold_tags = [], []
        previous_word_id = None
        for pred_id, word_id in zip(preds, word_ids):
            if word_id is None or word_id == previous_word_id:
                continue
            pred_tags.append(label_names[pred_id])
            gold_tags.append(label_names[example["pos_tags"][word_id]])
            previous_word_id = word_id

        outputs.append(pred_tags)
        targets.append(gold_tags)

    return score_pos(outputs, targets)


def bio_to_spans(tags: List[str]):
    spans = set()
    start = None
    label = None

    for i, tag in enumerate(tags):
        if tag == "O":
            if label is not None:
                spans.add((label, start, i - 1))
                start, label = None, None
            continue

        if "-" not in tag:
            if label is not None:
                spans.add((label, start, i - 1))
                start, label = None, None
            continue

        prefix, role = tag.split("-", 1)
        if prefix == "B":
            if label is not None:
                spans.add((label, start, i - 1))
            start, label = i, role
        elif prefix == "I":
            if label != role:
                if label is not None:
                    spans.add((label, start, i - 1))
                start, label = i, role
        else:
            if label is not None:
                spans.add((label, start, i - 1))
                start, label = None, None

    if label is not None:
        spans.add((label, start, len(tags) - 1))
    return spans


def flatten_conll_srl(split):
    flat = []
    for doc in split:
        for sent in doc["sentences"]:
            tokens = sent["words"]
            for frame in sent["srl_frames"]:
                labels = frame["frames"]
                if len(tokens) != len(labels):
                    continue
                try:
                    predicate_index = labels.index("B-V")
                except ValueError:
                    continue
                flat.append(
                    {
                        "tokens": tokens,
                        "predicate_index": predicate_index,
                        "labels": labels,
                    }
                )
    return flat


def preprocess_srl(example, tokenizer, label2id, max_length: int = 256):
    tokens = example["tokens"].copy()
    labels = example["labels"].copy()
    pred = example["predicate_index"]

    tokens = tokens[:pred] + ["[PRED]", tokens[pred], "[/PRED]"] + tokens[pred + 1 :]
    labels = labels[:pred] + ["O", labels[pred], "O"] + labels[pred + 1 :]

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    label_ids = []
    previous_word_id = None
    for word_id in encoding.word_ids():
        if word_id is None:
            label_ids.append(-100)
        elif word_id != previous_word_id:
            label_ids.append(label2id[labels[word_id]])
        else:
            label_ids.append(-100)
        previous_word_id = word_id

    encoding["labels"] = label_ids
    return encoding


def evaluate_srl(model, dataset, tokenizer, label_names: List[str]) -> Tuple[float, float, float]:
    model.eval()
    device = next(model.parameters()).device
    total_correct = total_pred = total_gold = 0

    for example in dataset:
        original_tokens = example["tokens"]
        pred_idx = example["predicate_index"]
        marked_tokens = (
            original_tokens[:pred_idx]
            + ["[PRED]", original_tokens[pred_idx], "[/PRED]"]
            + original_tokens[pred_idx + 1 :]
        )

        inputs = tokenizer(
            marked_tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(device)

        with torch.no_grad():
            preds = model(**inputs).logits.argmax(dim=-1)[0].cpu().tolist()

        pred_tags, gold_tags = [], []
        previous_word_id = None
        for pred_id, word_id in zip(preds, inputs.word_ids()):
            if word_id is None or word_id == previous_word_id:
                continue
            previous_word_id = word_id

            if marked_tokens[word_id] in ("[PRED]", "[/PRED]"):
                continue

            if word_id < pred_idx:
                gold_index = word_id
            elif word_id == pred_idx + 1:
                gold_index = pred_idx
            else:
                gold_index = word_id - 2

            if gold_index >= len(example["labels"]):
                continue
            pred_tags.append(label_names[pred_id])
            gold_tags.append(example["labels"][gold_index])

        pred_spans = bio_to_spans(pred_tags)
        gold_spans = bio_to_spans(gold_tags)
        total_correct += len(pred_spans & gold_spans)
        total_pred += len(pred_spans)
        total_gold += len(gold_spans)

    precision = total_correct / total_pred if total_pred else 0.0
    recall = total_correct / total_gold if total_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def make_token_model(model_name: str, num_labels: int, id2label=None, label2id=None):
    kwargs = {
        "num_labels": num_labels,
        "ignore_mismatched_sizes": True,
    }
    if id2label is not None:
        kwargs["id2label"] = id2label
    if label2id is not None:
        kwargs["label2id"] = label2id
    return AutoModelForTokenClassification.from_pretrained(model_name, **kwargs)


def run_pos(args, seed: int) -> Tuple[float, float, float]:
    seed_everything(seed)
    print(f"\n===== BERT POS | seed={seed} =====")
    raw = load_dataset("conll2000", trust_remote_code=True)

    # Keep the data split fixed across seeds so only training randomness changes.
    split = raw["train"].train_test_split(test_size=0.1, seed=args.split_seed)
    label_names = raw["train"].features["pos_tags"].feature.names
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    train_ds = split["train"].map(
        lambda x: preprocess_word_labels(x, tokenizer, "pos_tags"),
        batched=True,
        remove_columns=split["train"].column_names,
    )
    dev_ds = split["test"].map(
        lambda x: preprocess_word_labels(x, tokenizer, "pos_tags"),
        batched=True,
        remove_columns=split["test"].column_names,
    )

    model = make_token_model(args.model, len(label_names))
    with tempfile.TemporaryDirectory(prefix="bert_pos_") as tmpdir:
        trainer = Trainer(
            model=model,
            args=training_args(tmpdir, seed, args.epochs, args.batch, args.lr_token),
            train_dataset=train_ds,
            eval_dataset=dev_ds,
        )
        trainer.train()
        return evaluate_pos(model, raw["test"], tokenizer, label_names)


def run_ner_like(args, seed: int, dataset_name: str, task_name: str) -> Tuple[float, float, float]:
    seed_everything(seed)
    print(f"\n===== BERT {task_name.upper()} | seed={seed} =====")
    raw = load_dataset(dataset_name, trust_remote_code=True)
    label_names = raw["train"].features["ner_tags"].feature.names
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    train_ds = raw["train"].map(
        lambda x: preprocess_word_labels(x, tokenizer, "ner_tags"),
        batched=True,
        remove_columns=raw["train"].column_names,
    )
    dev_ds = raw["validation"].map(
        lambda x: preprocess_word_labels(x, tokenizer, "ner_tags"),
        batched=True,
        remove_columns=raw["validation"].column_names,
    )
    test_ds = raw["test"].map(
        lambda x: preprocess_word_labels(x, tokenizer, "ner_tags"),
        batched=True,
        remove_columns=raw["test"].column_names,
    )

    model = make_token_model(args.model, len(label_names))
    with tempfile.TemporaryDirectory(prefix=f"bert_{task_name}_") as tmpdir:
        trainer = Trainer(
            model=model,
            args=training_args(tmpdir, seed, args.epochs, args.batch, args.lr_token),
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            compute_metrics=ner_compute_metrics(label_names),
        )
        trainer.train()
        results = trainer.evaluate(eval_dataset=test_ds)

    return (
        float(results.get("eval_precision", 0.0)),
        float(results.get("eval_recall", 0.0)),
        float(results.get("eval_f1", 0.0)),
    )


def run_srl(args, seed: int) -> Tuple[float, float, float]:
    seed_everything(seed)
    print(f"\n===== BERT SRL | seed={seed} =====")
    raw = load_dataset(
        "ontonotes/conll2012_ontonotesv5",
        "english_v4",
        trust_remote_code=True,
    )
    train_data = flatten_conll_srl(raw["train"])
    dev_data = flatten_conll_srl(raw["validation"])
    test_data = flatten_conll_srl(raw["test"])

    all_labels = set()
    for split in (train_data, dev_data):
        for example in split:
            all_labels.update(example["labels"])
    label_names = sorted(all_labels)
    label2id = {label: i for i, label in enumerate(label_names)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[PRED]", "[/PRED]"]})

    train_raw = Dataset.from_list(train_data)
    dev_raw = Dataset.from_list(dev_data)
    train_ds = train_raw.map(
        lambda x: preprocess_srl(x, tokenizer, label2id),
        remove_columns=train_raw.column_names,
    )
    dev_ds = dev_raw.map(
        lambda x: preprocess_srl(x, tokenizer, label2id),
        remove_columns=dev_raw.column_names,
    )

    model = make_token_model(
        args.model,
        len(label_names),
        id2label=id2label,
        label2id=label2id,
    )
    model.resize_token_embeddings(len(tokenizer))

    with tempfile.TemporaryDirectory(prefix="bert_srl_") as tmpdir:
        trainer = Trainer(
            model=model,
            args=training_args(tmpdir, seed, args.epochs, args.batch, args.lr_srl),
            train_dataset=train_ds,
            eval_dataset=dev_ds,
        )
        trainer.train()
        return evaluate_srl(model, test_data, tokenizer, label_names)


def run_ontonotes5(args, seed: int) -> Tuple[float, float, float]:
    seed_everything(seed)
    print(f"\n===== BERT OntoNotes5 NER | seed={seed} =====")
    raw = load_dataset("tner/ontonotes5", trust_remote_code=True)
    label_names = [ONTONOTES_ID2LABEL[i] for i in range(len(ONTONOTES_ID2LABEL))]
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    train_ds = raw["train"].map(
        lambda x: preprocess_word_labels(x, tokenizer, "tags"),
        batched=True,
        remove_columns=raw["train"].column_names,
    )
    dev_ds = raw["validation"].map(
        lambda x: preprocess_word_labels(x, tokenizer, "tags"),
        batched=True,
        remove_columns=raw["validation"].column_names,
    )
    test_ds = raw["test"].map(
        lambda x: preprocess_word_labels(x, tokenizer, "tags"),
        batched=True,
        remove_columns=raw["test"].column_names,
    )

    model = make_token_model(
        args.model,
        len(label_names),
        id2label=ONTONOTES_ID2LABEL,
        label2id=ONTONOTES_LABEL2ID,
    )
    with tempfile.TemporaryDirectory(prefix="bert_ontonotes5_") as tmpdir:
        trainer = Trainer(
            model=model,
            args=training_args(tmpdir, seed, args.epochs, args.batch, args.lr_token),
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            compute_metrics=ner_compute_metrics(label_names),
        )
        trainer.train()
        results = trainer.evaluate(eval_dataset=test_ds)

    return (
        float(results.get("eval_precision", 0.0)),
        float(results.get("eval_recall", 0.0)),
        float(results.get("eval_f1", 0.0)),
    )


def run_task(args, task: str, seed: int) -> Tuple[float, float, float]:
    if task == "pos":
        return run_pos(args, seed)
    if task == "ner":
        return run_ner_like(args, seed, "conll2003", "ner")
    if task == "srl":
        return run_srl(args, seed)
    if task == "genia":
        return run_ner_like(args, seed, "chufangao/GENIA-NER", "genia")
    if task == "ontonotes5":
        return run_ontonotes5(args, seed)
    raise ValueError(f"Unsupported task: {task}")


def write_results_csv(rows: List[dict], out_path: str) -> None:
    fieldnames = ["model", "task", "run", "seed", "precision", "recall", "f1"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--task",
        default="all",
        choices=("all",) + TASKS,
        help="Run one task or all five tasks.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr-token", type=float, default=5e-5)
    parser.add_argument("--lr-srl", type=float, default=2e-5)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Fixed train/dev split seed for datasets without an official validation split.",
    )
    parser.add_argument(
        "--output",
        default="outputs/pr_bert_downstream_results.csv",
        help="CSV path. It is rewritten after every completed run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected_tasks = TASKS if args.task == "all" else (args.task,)
    rows = []
    for task in selected_tasks:
        for run_number, seed in enumerate(SEEDS, start=1):
            p, r, f1 = run_task(args, task, seed)
            row = {
                "model": args.model,
                "task": task,
                "run": run_number,
                "seed": seed,
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
            }
            rows.append(row)
            write_results_csv(rows, args.output)
            print(
                f"RESULT task={task} seed={seed}: "
                f"P={p:.4f} R={r:.4f} F1={f1:.4f}"
            )
            print(f"CSV updated: {args.output}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nFinished {len(rows)} runs. Results: {args.output}")


if __name__ == "__main__":
    main()