import argparse
import csv
import gc
import os
import random
import string
import tempfile
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

DEFAULT_MODEL = "./pr-gpt2-output"
SEEDS = (1, 2, 3)
#TASKS = ("pos", "ner", "srl", "genia", "ontonotes5")
TASKS = ("pos", "ner", "genia", "ontonotes5")
MAX_LENGTH = 256
GEN_MAX_NEW_TOKENS = 256

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
        remove_unused_columns=False,
        seed=seed,
        data_seed=seed,
        **mixed_precision_kwargs(),
    )


def make_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def make_model(model_name: str, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


def build_pos_prompt(tokens: List[str]) -> str:
    return f"Sentence: {' '.join(tokens)}\nPOS tags:"


def build_ner_prompt(tokens: List[str]) -> str:
    return f"Sentence: {' '.join(tokens)}\nNER tags:"


def build_srl_prompt(tokens: List[str], predicate_index: int) -> str:
    marked = (
        tokens[:predicate_index]
        + ["[PRED]", tokens[predicate_index], "[/PRED]"]
        + tokens[predicate_index + 1 :]
    )
    return f"Sentence: {' '.join(marked)}\nSRL tags:"


def preprocess_prompt_completion(prompt: str, gold_tags: List[str], tokenizer, max_length: int = MAX_LENGTH):
    full_text = prompt + " " + " ".join(gold_tags)
    tokenized = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        padding=False,
        max_length=max_length,
    )["input_ids"]

    labels = tokenized["input_ids"].copy()
    prompt_len = min(len(prompt_ids), max_length)
    labels[:prompt_len] = [-100] * prompt_len
    labels = [label if mask == 1 else -100 for label, mask in zip(labels, tokenized["attention_mask"])]
    tokenized["labels"] = labels
    return tokenized


def preprocess_pos(example, tokenizer, label_names: List[str]):
    gold_tags = [label_names[i] for i in example["pos_tags"]]
    return preprocess_prompt_completion(build_pos_prompt(example["tokens"]), gold_tags, tokenizer)


def preprocess_ner(example, tokenizer, label_names: List[str]):
    gold_tags = [label_names[i] for i in example["ner_tags"]]
    return preprocess_prompt_completion(build_ner_prompt(example["tokens"]), gold_tags, tokenizer)


def preprocess_ontonotes(example, tokenizer):
    gold_tags = [ONTONOTES_ID2LABEL[i] for i in example["tags"]]
    return preprocess_prompt_completion(build_ner_prompt(example["tokens"]), gold_tags, tokenizer)


def preprocess_srl(example, tokenizer):
    prompt = build_srl_prompt(example["tokens"], example["predicate_index"])
    return preprocess_prompt_completion(prompt, example["labels"], tokenizer)


def prf1(num_correct: int, num_attempted: int, num_gold: int) -> Tuple[float, float, float]:
    precision = num_correct / num_attempted if num_attempted else 0.0
    recall = num_correct / num_gold if num_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score_pos(outputs: Iterable[List[str]], targets: Iterable[List[str]]) -> Tuple[float, float, float]:
    num_correct = num_attempted = num_gold = 0
    for pred_tags, gold_tags in zip(outputs, targets):
        num_attempted += len(pred_tags)
        num_gold += len(gold_tags)
        num_correct += sum(p == g for p, g in zip(pred_tags, gold_tags))
    return prf1(num_correct, num_attempted, num_gold)


def clean_tag(token: str) -> str:
    return (token or "").strip().strip(string.punctuation)


def sanitize_to_label_set(pred_tags: List[str], label_set: set) -> List[str]:
    cleaned = []
    for tag in pred_tags:
        tag = clean_tag(tag)
        cleaned.append(tag if tag in label_set else "O")
    return cleaned


def generated_suffix(model, tokenizer, prompt: str, delimiter: str, max_new_tokens: int):
    device = next(model.parameters()).device
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    if delimiter in decoded:
        return decoded.split(delimiter, 1)[1].strip()
    return ""


def evaluate_pos(model, dataset, tokenizer, label_names: List[str]) -> Tuple[float, float, float]:
    model.eval()
    outputs, targets = [], []

    for example in dataset:
        gold_tags = [label_names[i] for i in example["pos_tags"]]
        pred_text = generated_suffix(
            model,
            tokenizer,
            build_pos_prompt(example["tokens"]),
            "POS tags:",
            min(GEN_MAX_NEW_TOKENS, len(gold_tags) + 20),
        )
        pred_tags = pred_text.split()[: len(gold_tags)]
        if len(pred_tags) < len(gold_tags):
            pred_tags += [""] * (len(gold_tags) - len(pred_tags))

        outputs.append(pred_tags)
        targets.append(gold_tags)

    return score_pos(outputs, targets)


def evaluate_ner(model, dataset, tokenizer, label_names: List[str], tag_column: str) -> Tuple[float, float, float]:
    model.eval()
    label_set = set(label_names)
    gold_seqs, pred_seqs = [], []

    for example in dataset:
        gold_tags = [label_names[i] for i in example[tag_column]]
        pred_text = generated_suffix(
            model,
            tokenizer,
            build_ner_prompt(example["tokens"]),
            "NER tags:",
            min(GEN_MAX_NEW_TOKENS, len(gold_tags) + 20),
        )
        pred_tags = sanitize_to_label_set(pred_text.split(), label_set)
        if len(pred_tags) < len(gold_tags):
            pred_tags += ["O"] * (len(gold_tags) - len(pred_tags))
        pred_tags = pred_tags[: len(gold_tags)]

        gold_seqs.append(gold_tags)
        pred_seqs.append(pred_tags)

    return (
        precision_score(gold_seqs, pred_seqs),
        recall_score(gold_seqs, pred_seqs),
        f1_score(gold_seqs, pred_seqs),
    )


def evaluate_ontonotes(model, dataset, tokenizer) -> Tuple[float, float, float]:
    label_names = [ONTONOTES_ID2LABEL[i] for i in range(len(ONTONOTES_ID2LABEL))]
    model.eval()
    label_set = set(label_names)
    gold_seqs, pred_seqs = [], []

    for example in dataset:
        gold_tags = [ONTONOTES_ID2LABEL[i] for i in example["tags"]]
        pred_text = generated_suffix(
            model,
            tokenizer,
            build_ner_prompt(example["tokens"]),
            "NER tags:",
            min(GEN_MAX_NEW_TOKENS, len(gold_tags) + 20),
        )
        pred_tags = sanitize_to_label_set(pred_text.split(), label_set)
        if len(pred_tags) < len(gold_tags):
            pred_tags += ["O"] * (len(gold_tags) - len(pred_tags))
        pred_tags = pred_tags[: len(gold_tags)]

        gold_seqs.append(gold_tags)
        pred_seqs.append(pred_tags)

    return (
        precision_score(gold_seqs, pred_seqs),
        recall_score(gold_seqs, pred_seqs),
        f1_score(gold_seqs, pred_seqs),
    )


def bio_to_spans(tags: List[str]):
    spans = set()
    start = None
    label = None

    for i, tag in enumerate(tags):
        if tag == "O" or "-" not in tag:
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


def evaluate_srl(model, dataset, tokenizer, label_names: List[str]) -> Tuple[float, float, float]:
    model.eval()
    label_set = set(label_names)
    total_correct = total_pred = total_gold = 0

    for example in dataset:
        gold_tags = example["labels"]
        prompt = build_srl_prompt(example["tokens"], example["predicate_index"])
        pred_text = generated_suffix(
            model,
            tokenizer,
            prompt,
            "SRL tags:",
            min(GEN_MAX_NEW_TOKENS, len(gold_tags) + 20),
        )
        pred_tags = sanitize_to_label_set(pred_text.split(), label_set)
        if len(pred_tags) < len(gold_tags):
            pred_tags += ["O"] * (len(gold_tags) - len(pred_tags))
        pred_tags = pred_tags[: len(gold_tags)]

        pred_spans = bio_to_spans(pred_tags)
        gold_spans = bio_to_spans(gold_tags)
        total_correct += len(pred_spans & gold_spans)
        total_pred += len(pred_spans)
        total_gold += len(gold_spans)

    precision = total_correct / total_pred if total_pred else 0.0
    recall = total_correct / total_gold if total_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def train_causal_lm(args, seed: int, train_ds, dev_ds, tokenizer, prefix: str):
    model = make_model(args.model, tokenizer)
    with tempfile.TemporaryDirectory(prefix=prefix) as tmpdir:
        trainer = Trainer(
            model=model,
            args=training_args(tmpdir, seed, args.epochs, args.batch, args.lr),
            train_dataset=train_ds,
            eval_dataset=dev_ds,
        )
        trainer.train()
    return model


def run_pos(args, seed: int) -> Tuple[float, float, float]:
    seed_everything(seed)
    print(f"\n===== GPT-2 POS | seed={seed} =====")
    raw = load_dataset("conll2000", trust_remote_code=True)
    split = raw["train"].train_test_split(test_size=0.1, seed=args.split_seed)
    label_names = raw["train"].features["pos_tags"].feature.names
    tokenizer = make_tokenizer(args.model)

    train_ds = split["train"].map(
        lambda x: preprocess_pos(x, tokenizer, label_names),
        remove_columns=split["train"].column_names,
    )
    dev_ds = split["test"].map(
        lambda x: preprocess_pos(x, tokenizer, label_names),
        remove_columns=split["test"].column_names,
    )

    model = train_causal_lm(args, seed, train_ds, dev_ds, tokenizer, "gpt2_pos_")
    return evaluate_pos(model, raw["test"], tokenizer, label_names)


def run_ner_like(args, seed: int, dataset_name: str, task_name: str) -> Tuple[float, float, float]:
    seed_everything(seed)
    print(f"\n===== GPT-2 {task_name.upper()} | seed={seed} =====")
    raw = load_dataset(dataset_name, trust_remote_code=True)
    label_names = raw["train"].features["ner_tags"].feature.names
    tokenizer = make_tokenizer(args.model)

    # Use the official validation split, fixed across seeds.
    train_ds = raw["train"].map(
        lambda x: preprocess_ner(x, tokenizer, label_names),
        remove_columns=raw["train"].column_names,
    )
    dev_ds = raw["validation"].map(
        lambda x: preprocess_ner(x, tokenizer, label_names),
        remove_columns=raw["validation"].column_names,
    )

    model = train_causal_lm(args, seed, train_ds, dev_ds, tokenizer, f"gpt2_{task_name}_")
    return evaluate_ner(model, raw["test"], tokenizer, label_names, "ner_tags")


def run_srl(args, seed: int) -> Tuple[float, float, float]:
    seed_everything(seed)
    print(f"\n===== GPT-2 SRL | seed={seed} =====")
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

    tokenizer = make_tokenizer(args.model)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[PRED]", "[/PRED]"]})

    train_raw = Dataset.from_list(train_data)
    dev_raw = Dataset.from_list(dev_data)
    train_ds = train_raw.map(
        lambda x: preprocess_srl(x, tokenizer),
        remove_columns=train_raw.column_names,
    )
    dev_ds = dev_raw.map(
        lambda x: preprocess_srl(x, tokenizer),
        remove_columns=dev_raw.column_names,
    )

    model = make_model(args.model, tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    with tempfile.TemporaryDirectory(prefix="gpt2_srl_") as tmpdir:
        trainer = Trainer(
            model=model,
            args=training_args(tmpdir, seed, args.epochs, args.batch, args.lr),
            train_dataset=train_ds,
            eval_dataset=dev_ds,
        )
        trainer.train()

    return evaluate_srl(model, test_data, tokenizer, label_names)


def run_ontonotes5(args, seed: int) -> Tuple[float, float, float]:
    seed_everything(seed)
    print(f"\n===== GPT-2 OntoNotes5 NER | seed={seed} =====")
    raw = load_dataset("tner/ontonotes5", trust_remote_code=True)
    tokenizer = make_tokenizer(args.model)

    train_ds = raw["train"].map(
        lambda x: preprocess_ontonotes(x, tokenizer),
        remove_columns=raw["train"].column_names,
    )
    dev_ds = raw["validation"].map(
        lambda x: preprocess_ontonotes(x, tokenizer),
        remove_columns=raw["validation"].column_names,
    )

    model = train_causal_lm(args, seed, train_ds, dev_ds, tokenizer, "gpt2_ontonotes5_")
    return evaluate_ontonotes(model, raw["test"], tokenizer)


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
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Fixed train/dev split seed for datasets without an official validation split.",
    )
    parser.add_argument(
        "--output",
        default="outputs/pr_gpt2_downstream_results.csv",
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