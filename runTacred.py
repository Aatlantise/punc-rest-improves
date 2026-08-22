import argparse
import json
import os
import random 
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import csv

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GPT2ForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
def seed_everything(seed: int):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# taken from scorer.py from tacred folder
NO_RELATION = "no_relation"
def score(key, prediction, verbose=False):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation    = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]
         
        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold    = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro   = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro




def load_json_list(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}, got {type(data)}")
    return data


def read_gold_labels(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def build_label_maps(train_data: List[dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = sorted({ex["relation"] for ex in train_data})
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


def insert_entity_markers(tokens: List[str], s0: int, s1: int, o0: int, o1: int) -> str:
    out = []
    for i, tok in enumerate(tokens):
        if i == s0:
            out.append("<SUBJ>")
        if i == o0:
            out.append("<OBJ>")
        out.append(tok)
        if i == s1:
            out.append("</SUBJ>")
        if i == o1:
            out.append("</OBJ>")
    return " ".join(out)


@dataclass
class EncodedDataset(torch.utils.data.Dataset):
    encodings: Dict[str, List[List[int]]]
    labels: List[int]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def encode_split(data: List[dict], tokenizer, label2id: Dict[str, int]) -> EncodedDataset:
    texts, y = [], []
    for ex in data:
        txt = insert_entity_markers(
            ex["token"],
            ex["subj_start"],
            ex["subj_end"],
            ex["obj_start"],
            ex["obj_end"],
        )
        texts.append(txt)
        y.append(label2id[ex["relation"]])

    enc = tokenizer(texts, truncation=True, max_length=256, padding=False)
    return EncodedDataset(encodings=enc, labels=y)



def build_model_and_tokenizer(model_name: str, num_labels: int, label2id: Dict[str, int], id2label: Dict[int, str]):
    if model_name.lower() == "gpt2" or "gpt2" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = GPT2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            pad_token_id=tokenizer.pad_token_id,
        )
        model.config.label2id = label2id
        model.config.id2label = id2label
        return model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    return model, tokenizer






def run_tacred(args, seed: int) -> Tuple[float, float, float]:
    seed_everything(seed)

    # fixed paths (no extra args)
    train_json = "tacred/data/json/train.json"
    dev_json = "tacred/data/json/dev.json"
    test_json = "tacred/data/json/test.json"
    test_gold = "tacred/data/gold/test.gold"

    train_data = load_json_list(train_json)
    dev_data = load_json_list(dev_json)
    test_data = load_json_list(test_json)
    gold = read_gold_labels(test_gold)

    label2id, id2label = build_label_maps(train_data)
    num_labels = len(label2id)

    model, tokenizer = build_model_and_tokenizer(args.model, num_labels, label2id, id2label)

    # add marker tokens (helps for BERT; harmless for GPT-2)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<SUBJ>", "</SUBJ>", "<OBJ>", "</OBJ>"]})
    model.resize_token_embeddings(len(tokenizer))

    train_ds = encode_split(train_data, tokenizer, label2id)
    dev_ds = encode_split(dev_data, tokenizer, label2id)
    test_ds = encode_split(test_data, tokenizer, label2id)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # fixed hyperparams (no CLI args)
    run_dir = os.path.join("outputs", "checkpoints", args.model.replace("/", "_"), f"seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=run_dir,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=100,
        report_to="none",
        seed=seed,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
    )

    trainer.train()

    # predict on test (no files)
    pred = trainer.predict(test_ds)
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pred_labels = [id2label[int(i)] for i in pred_ids.tolist()]

    if len(pred_labels) != len(gold):
        raise RuntimeError(
            f"Gold/pred length mismatch: gold={len(gold)} pred={len(pred_labels)}. "
            f"Check data/gold/test.gold aligns with data/json/test.json."
        )

    p, r, f1 = score(gold, pred_labels, verbose=False)
    return float(p), float(r), float(f1)

def write_results_csv(rows, out_path):
    fieldnames = ["model", "task", "run", "seed", "precision", "recall", "f1"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-uncased",
        help="what model to fine-tune (bert-base-uncased, roberta-base, gpt2, ...)",
    )
    # task is tacred
    parser.add_argument("--task", type=str, default="tacred")
    args = parser.parse_args()

    results = []
    for i in range(3):
        if args.task == "tacred":
            p, r, f1 = run_tacred(args, i)
        else:
            p, r, f1 = 0.0, 0.0, 0.0
            print("task not available")

        results.append(
            {
                "model": args.model,
                "task": args.task,
                "run": i,
                "seed": i,
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
            }
        )
        print(f"[run {i}] P={p:.3%} R={r:.3%} F1={f1:.3%}")

    save_csv = os.path.join("outputs", "generated", f"{args.model.replace('/', '_')}_{args.task}.csv")
    os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)

    write_results_csv(results, save_csv)
    print(f"\nSaved results to {save_csv}\n")