import argparse
import csv
import gc
import inspect
import os
import re
import statistics
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


WEIGHT_FILENAMES = {
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}

RUN_COLUMNS = [
    "row_type",
    "model_key",
    "architecture",
    "variant",
    "model_source",
    "resolved_model_path",
    "dataset",
    "seed",
    "num_labels",
    "train_size",
    "validation_size",
    "test_size",
    "epochs",
    "learning_rate",
    "train_batch_size",
    "eval_batch_size",
    "max_length",
    "weight_decay",
    "warmup_ratio",
    "test_accuracy",
    "test_macro_f1",
    "test_loss",
    "train_runtime_sec",
    "eval_runtime_sec",
    "timestamp",
]

LEARNING_RATE = {
    "bert": 2e-5,
    "gpt2": 1e-5,
    "t5": 3e-4,
}

def parse_args():
    p = argparse.ArgumentParser()

    # Existing PR checkpoints.
    p.add_argument("--gpt2-pr", default="./pr-gpt2-output")
    p.add_argument("--bert-pr", default="./pr-bert-financial-news-continued")
    p.add_argument("--t5-pr", default="./T5_accumulation")

    p.add_argument(
        "--models",
        default="all",
        help=(
            "Comma-separated subset of: "
            "gpt2-medium,gpt2-pr,bert-base,bert-pr,t5-base,t5-pr; default=all"
        ),
    )
    p.add_argument(
        "--datasets",
        default="clinc150,banking77",
        help="Comma-separated subset of: clinc150,banking77",
    )
    p.add_argument("--seeds", default="1,2,3")

    p.add_argument("--output-csv", default="intent_results.csv")
    p.add_argument(
        "--summary-csv",
        default=None,
        help="Default: <output-csv stem>_summary.csv",
    )

    # Intent utterances are short, so 128 is ample and keeps the runs fast.
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--epochs", type=float, default=5.0)
    p.add_argument("--train-batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--dataloader-num-workers", type=int, default=2)

    p.add_argument(
        "--precision",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="bf16 is a good default on H100.",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run combinations already present in output CSV.",
    )
    return p.parse_args()


def contains_weights(path: Path) -> bool:
    return any((path / name).exists() for name in WEIGHT_FILENAMES)


def checkpoint_number(path: Path):
    m = re.search(r"checkpoint-(\d+)$", path.name)
    return int(m.group(1)) if m else None


def resolve_local_checkpoint(path_string: str) -> str:
    """
    Accept either:
      1) a normal HF model directory containing weights, or
      2) a parent directory containing checkpoint-* subdirectories.

    If a parent directory has multiple checkpoint-* folders, select the one
    with the largest checkpoint number.
    """
    path = Path(path_string).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"PR checkpoint path does not exist: {path}")

    if path.is_file():
        raise ValueError(
            f"Expected a Hugging Face model DIRECTORY, but got file: {path}"
        )

    if contains_weights(path):
        return str(path)

    candidates = []
    for child in path.rglob("*"):
        if child.is_dir() and contains_weights(child):
            candidates.append(child)

    if not candidates:
        visible = sorted(p.name for p in path.iterdir())[:30]
        raise FileNotFoundError(
            f"Found directory {path}, but no Hugging Face model weights inside it.\n"
            f"Expected one of: {sorted(WEIGHT_FILENAMES)}\n"
            f"Top-level contents: {visible}"
        )

    numbered = [(checkpoint_number(c), c) for c in candidates]
    numbered = [(n, c) for n, c in numbered if n is not None]
    if numbered:
        chosen = max(numbered, key=lambda x: x[0])[1]
    else:
        chosen = max(candidates, key=lambda c: c.stat().st_mtime)

    print(f"[checkpoint resolver] {path} -> {chosen}")
    return str(chosen)


def model_specs(args):
    specs = {
        "gpt2-medium": {
            "architecture": "gpt2",
            "variant": "base",
            "source": "gpt2-medium",
            "tokenizer": "gpt2-medium",
            "local": False,
        },
        "gpt2-pr": {
            "architecture": "gpt2",
            "variant": "pr",
            "source": args.gpt2_pr,
            "tokenizer": "gpt2-medium",
            "local": True,
        },
        "bert-base": {
            "architecture": "bert",
            "variant": "base",
            "source": "bert-base-uncased",
            "tokenizer": "bert-base-uncased",
            "local": False,
        },
        "bert-pr": {
            "architecture": "bert",
            "variant": "pr",
            "source": args.bert_pr,
            "tokenizer": "bert-base-uncased",
            "local": True,
        },
        "t5-base": {
            "architecture": "t5",
            "variant": "base",
            "source": "t5-base",
            "tokenizer": "t5-base",
            "local": False,
        },
        "t5-pr": {
            "architecture": "t5",
            "variant": "pr",
            "source": args.t5_pr,
            "tokenizer": "t5-base",
            "local": True,
        },
    }
    return specs


def normalize_clinc_labels(ds: DatasetDict) -> DatasetDict:
    """
    DeepPavlov/clinc_oos currently exposes integer `label` plus `label_text`.
    Older mirrors may expose `intent`. Normalize to one integer column: label.
    """
    out = DatasetDict()
    for split, data in ds.items():
        cols = data.column_names

        if "label" in cols:
            out[split] = data
        elif "intent" in cols:
            # If intent is already a ClassLabel/int, renaming is enough.
            out[split] = data.rename_column("intent", "label")
        else:
            raise ValueError(
                f"CLINC150 split {split} has neither 'label' nor 'intent': {cols}"
            )

    # Make sure labels are contiguous 0..N-1 even if a future mirror changes IDs.
    all_labels = []
    for split in out:
        all_labels.extend(out[split]["label"])

    # If labels are strings, map them. If they are ints, preserve sorted mapping.
    unique = sorted(set(all_labels))
    label2id = {lab: i for i, lab in enumerate(unique)}

    if unique != list(range(len(unique))):
        def remap(example):
            example["label"] = label2id[example["label"]]
            return example
        out = out.map(remap)

    return out


def load_intent_dataset(name: str):
    if name == "clinc150":
        # Standard OOS+ setting: 15,250 train / 3,100 val / 5,500 test,
        # with 150 in-scope intents + OOS = 151 labels.
        ds = load_dataset("DeepPavlov/clinc_oos", "plus")
        ds = normalize_clinc_labels(ds)

    elif name == "banking77":
        # BANKING77 provides train/test but no validation split.
        ds0 = load_dataset("PolyAI/banking77")
        # FIXED validation split: identical for seeds 1,2,3 and every model.
        split = ds0["train"].train_test_split(
            test_size=0.10,
            seed=42,
            stratify_by_column="label",
        )
        ds = DatasetDict(
            train=split["train"],
            validation=split["test"],
            test=ds0["test"],
        )

    else:
        raise ValueError(f"Unknown dataset: {name}")

    labels = set()
    for split in ds:
        labels.update(ds[split]["label"])
    num_labels = len(labels)

    expected = {"clinc150": 151, "banking77": 77}[name]
    if num_labels != expected:
        raise ValueError(
            f"{name}: expected {expected} labels, found {num_labels}."
        )

    return ds, num_labels


def label_maps(ds: DatasetDict, num_labels: int):
    """
    Prefer human-readable label names when available, otherwise use string IDs.
    """
    id2label = {i: str(i) for i in range(num_labels)}

    train = ds["train"]
    if "label_text" in train.column_names:
        pairs = {}
        for label, text in zip(train["label"], train["label_text"]):
            pairs[int(label)] = str(text)
        if len(pairs) == num_labels:
            id2label = {i: pairs[i] for i in range(num_labels)}
    else:
        feature = train.features.get("label")
        if hasattr(feature, "names") and feature.names:
            id2label = {i: feature.names[i] for i in range(num_labels)}

    label2id = {name: i for i, name in id2label.items()}
    return id2label, label2id


def tokenize_dataset(ds, tokenizer, max_length):
    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    result = DatasetDict()
    for split, data in ds.items():
        remove_cols = [c for c in data.column_names if c != "label"]
        result[split] = data.map(
            tok,
            batched=True,
            remove_columns=remove_cols,
            desc=f"Tokenizing {split}",
        )
    return result


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Some seq2seq-style model outputs can arrive as a tuple.
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def make_training_args(args, output_dir, seed, learning_rate):
    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=True,
    )

    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in sig:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    if args.precision == "bf16":
        kwargs["bf16"] = True
    elif args.precision == "fp16":
        kwargs["fp16"] = True

    return TrainingArguments(**kwargs)


def load_completed(output_csv):
    completed = set()
    p = Path(output_csv)
    if not p.exists():
        return completed

    with p.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("row_type", "run") != "run":
                continue
            try:
                completed.add(
                    (row["model_key"], row["dataset"], int(row["seed"]))
                )
            except (KeyError, ValueError):
                pass
    return completed


def append_row(output_csv, row):
    p = Path(output_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists() or p.stat().st_size == 0

    with p.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RUN_COLUMNS})


def read_run_rows(output_csv):
    p = Path(output_csv)
    if not p.exists():
        return []
    with p.open(newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("row_type", "run") == "run"]


def write_summary(output_csv, summary_csv):
    rows = read_run_rows(output_csv)
    grouped = {}

    for r in rows:
        key = (r["model_key"], r["architecture"], r["variant"], r["dataset"])
        grouped.setdefault(key, []).append(r)

    summary_fields = [
        "model_key",
        "architecture",
        "variant",
        "dataset",
        "n_seeds",
        "seeds",
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
    ]

    out_path = Path(summary_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()

        for key in sorted(grouped):
            model_key, architecture, variant, dataset = key
            group = sorted(grouped[key], key=lambda x: int(x["seed"]))
            acc = [float(x["test_accuracy"]) for x in group]
            f1 = [float(x["test_macro_f1"]) for x in group]
            seeds = [int(x["seed"]) for x in group]

            writer.writerow(
                {
                    "model_key": model_key,
                    "architecture": architecture,
                    "variant": variant,
                    "dataset": dataset,
                    "n_seeds": len(group),
                    "seeds": ",".join(map(str, seeds)),
                    "accuracy_mean": statistics.mean(acc),
                    "accuracy_std": statistics.stdev(acc) if len(acc) > 1 else 0.0,
                    "macro_f1_mean": statistics.mean(f1),
                    "macro_f1_std": statistics.stdev(f1) if len(f1) > 1 else 0.0,
                }
            )


def train_one(args, model_key, spec, dataset_name, seed):
    print("\n" + "=" * 90)
    print(f"MODEL={model_key}  DATASET={dataset_name}  SEED={seed}")
    print("=" * 90)

    set_seed(seed)

    if torch.cuda.is_available():
        # Good speedup on Ampere/Hopper; does not change the benchmark definition.
        torch.backends.cuda.matmul.allow_tf32 = True

    model_path = (
        resolve_local_checkpoint(spec["source"])
        if spec["local"]
        else spec["source"]
    )

    ds, num_labels = load_intent_dataset(dataset_name)
    id2label, label2id = label_maps(ds, num_labels)

    # Use the ORIGINAL/base tokenizer for PR checkpoints. Continued pretraining
    # did not change the vocabulary, and PR folders do not need tokenizer files.
    tokenizer = AutoTokenizer.from_pretrained(spec["tokenizer"])

    # GPT-2 has no pad token by default. EOS-as-pad is the standard simple fix
    # for sequence classification.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = tokenize_dataset(ds, tokenizer, args.max_length)

    # This loads the pretrained backbone weights and initializes a fresh
    # classification head for this downstream run.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Avoid KV-cache warnings / wasted memory while fine-tuning GPT-2/T5.
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    with tempfile.TemporaryDirectory(prefix="intent_bench_") as tmpdir:
        lr = LEARNING_RATE[spec["architecture"]]
        training_args = make_training_args(args, tmpdir, seed, lr)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        train_output = trainer.train()
        test_metrics = trainer.evaluate(
            eval_dataset=tokenized["test"],
            metric_key_prefix="test",
        )

    row = {
        "row_type": "run",
        "model_key": model_key,
        "architecture": spec["architecture"],
        "variant": spec["variant"],
        "model_source": spec["source"],
        "resolved_model_path": model_path,
        "dataset": dataset_name,
        "seed": seed,
        "num_labels": num_labels,
        "train_size": len(ds["train"]),
        "validation_size": len(ds["validation"]),
        "test_size": len(ds["test"]),
        "epochs": args.epochs,
        "learning_rate": lr,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_length": args.max_length,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "test_accuracy": test_metrics["test_accuracy"],
        "test_macro_f1": test_metrics["test_macro_f1"],
        "test_loss": test_metrics.get("test_loss", ""),
        "train_runtime_sec": train_output.metrics.get("train_runtime", ""),
        "eval_runtime_sec": test_metrics.get("test_runtime", ""),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    print(
        f"RESULT {model_key} / {dataset_name} / seed {seed}: "
        f"accuracy={row['test_accuracy']:.6f}, "
        f"macro_f1={row['test_macro_f1']:.6f}"
    )

    del trainer, model, tokenized, tokenizer, ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


def main():
    args = parse_args()

    output_csv = str(Path(args.output_csv))
    if args.summary_csv is None:
        p = Path(output_csv)
        summary_csv = str(p.with_name(f"{p.stem}_summary{p.suffix or '.csv'}"))
    else:
        summary_csv = args.summary_csv

    all_specs = model_specs(args)

    if args.models.strip().lower() == "all":
        selected_models = list(all_specs.keys())
    else:
        selected_models = [x.strip() for x in args.models.split(",") if x.strip()]

    selected_datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    bad_models = [x for x in selected_models if x not in all_specs]
    if bad_models:
        raise ValueError(f"Unknown model keys: {bad_models}")

    bad_datasets = [x for x in selected_datasets if x not in {"clinc150", "banking77"}]
    if bad_datasets:
        raise ValueError(f"Unknown datasets: {bad_datasets}")

    # Resolve PR paths up front so a typo fails before a long run starts.
    for model_key in selected_models:
        spec = all_specs[model_key]
        if spec["local"]:
            resolved = resolve_local_checkpoint(spec["source"])
            print(f"{model_key}: {resolved}")

    completed = set() if args.no_resume else load_completed(output_csv)

    total = len(selected_models) * len(selected_datasets) * len(seeds)
    done = 0

    print(f"\nPlanned runs: {total}")
    print(f"Per-seed results: {output_csv}")
    print(f"Summary:          {summary_csv}")
    print("Downstream checkpoints: NOT SAVED\n")

    for model_key in selected_models:
        spec = all_specs[model_key]
        for dataset_name in selected_datasets:
            for seed in seeds:
                done += 1
                key = (model_key, dataset_name, seed)

                if key in completed:
                    print(f"[{done}/{total}] SKIP completed: {key}")
                    continue

                print(f"[{done}/{total}] START: {key}")
                row = train_one(args, model_key, spec, dataset_name, seed)
                append_row(output_csv, row)

                # Regenerate summary after every successful run, so partial jobs
                # still leave a useful summary on disk.
                write_summary(output_csv, summary_csv)

    write_summary(output_csv, summary_csv)

    print("\nFinished.")
    print(f"Raw results: {output_csv}")
    print(f"Summary:     {summary_csv}")


if __name__ == "__main__":
    main()