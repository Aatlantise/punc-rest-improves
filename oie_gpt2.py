from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import re
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import auc
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EncoderDecoderModel,
    get_linear_schedule_with_warmup,
)


WEIGHT_FILENAMES = {
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}


# ---------------------------------------------------------------------------
# Reproducibility and model-path helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def safe_name(value: str) -> str:
    value = value.rstrip("/")
    name = os.path.basename(value) or value
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name)


def contains_weights(path: Path) -> bool:
    return path.is_dir() and any((path / name).exists() for name in WEIGHT_FILENAMES)


def checkpoint_number(path: Path) -> int | None:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else None


def resolve_model_path(model_string: str) -> str:
    """
    Resolve:
      * a Hugging Face model id, e.g. bert-base-uncased / gpt2-medium
      * a local HF model directory
      * a parent directory containing checkpoint-* model directories
    """
    path = Path(model_string).expanduser()

    if not path.exists():
        # Simple names are assumed to be Hugging Face model IDs.
        if model_string.startswith(".") or model_string.startswith("/"):
            raise FileNotFoundError(f"Model path does not exist: {model_string}")
        if model_string.startswith("checkpoints/"):
            raise FileNotFoundError(f"Model path does not exist: {model_string}")
        return model_string

    path = path.resolve()
    if path.is_file():
        raise ValueError(f"Expected a model directory, got file: {path}")

    if contains_weights(path):
        return str(path)

    candidates = [p for p in path.rglob("*") if contains_weights(p)]
    if not candidates:
        visible = sorted(p.name for p in path.iterdir())[:30]
        raise FileNotFoundError(
            f"Found directory {path}, but no Hugging Face model weights inside it.\n"
            f"Expected one of: {sorted(WEIGHT_FILENAMES)}\n"
            f"Top-level contents: {visible}"
        )

    numbered = [(checkpoint_number(p), p) for p in candidates]
    numbered = [(n, p) for n, p in numbered if n is not None]

    if numbered:
        chosen = max(numbered, key=lambda item: item[0])[1]
    else:
        chosen = max(candidates, key=lambda p: p.stat().st_mtime)

    print(f"[checkpoint resolver] {path} -> {chosen}")
    return str(chosen)


# ---------------------------------------------------------------------------
# Official OIE2016 split reading
# ---------------------------------------------------------------------------

def serialize_extractions(extractions: list[dict[str, Any]]) -> str:
    """
    Serialize one sentence's gold extractions as:
        (ARG1; REL; ARG2; ARG3; ...) || (ARG1; REL; ARG2; ...)
    """
    pieces: list[str] = []

    for extraction in extractions:
        predicate = extraction["predicate"].strip()
        args = [arg.strip() for arg in extraction["args"] if arg.strip()]

        if not predicate or not args:
            continue

        fields = [args[0], predicate] + args[1:]
        pieces.append("(" + "; ".join(fields) + ")")

    return " || ".join(pieces)


def read_oie_split(path: str | Path) -> list[dict[str, Any]]:
    """
    Read an official .oie file.

    GoldReader in oie-benchmark interprets each line as:
        sentence<TAB>base_relation<TAB>full_relation<TAB>arg1<TAB>arg2...

    We preserve all arguments and group all lines belonging to the same
    sentence, while preserving first-seen sentence order.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing OIE split: {path}")

    grouped: dict[str, list[dict[str, Any]]] = {}
    sentence_order: list[str] = []

    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue

            fields = line.split("\t")
            if len(fields) < 4:
                raise ValueError(
                    f"{path}:{line_number}: expected at least 4 tab-separated "
                    f"fields, found {len(fields)}"
                )

            sentence = fields[0].strip()
            predicate = fields[2].strip()
            args = [field.strip() for field in fields[3:]]

            if not sentence or not predicate:
                continue

            # Match GoldReader's behavior: arguments are whatever follows
            # the first three columns.  Empty fields are removed only from
            # our generation target because they cannot carry lexical content.
            args = [arg for arg in args if arg]
            if not args:
                continue

            if sentence not in grouped:
                grouped[sentence] = []
                sentence_order.append(sentence)

            grouped[sentence].append(
                {
                    "predicate": predicate,
                    "args": args,
                }
            )

    rows: list[dict[str, Any]] = []
    for sentence in sentence_order:
        extractions = grouped[sentence]
        target = serialize_extractions(extractions)
        if target:
            rows.append(
                {
                    "source": sentence,
                    "target": target,
                    "extractions": extractions,
                }
            )

    return rows


def load_official_splits(data_dir: str | Path):
    data_dir = Path(data_dir)
    train_path = data_dir / "train.oie"
    dev_path = data_dir / "dev.oie"
    test_path = data_dir / "test.oie"

    missing = [str(p) for p in (train_path, dev_path, test_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing official OIE2016 split files. Expected train.oie, dev.oie, "
            f"and test.oie under {data_dir}. Missing: {missing}"
        )

    train_rows = read_oie_split(train_path)
    dev_rows = read_oie_split(dev_path)
    test_rows = read_oie_split(test_path)

    if not train_rows or not dev_rows or not test_rows:
        raise ValueError(
            "At least one split is empty after parsing: "
            f"train={len(train_rows)}, dev={len(dev_rows)}, test={len(test_rows)}"
        )

    print(
        "Official OIE2016 sentence counts: "
        f"train={len(train_rows)}, dev={len(dev_rows)}, test={len(test_rows)}"
    )
    print(
        "Official OIE2016 extraction counts: "
        f"train={sum(len(r['extractions']) for r in train_rows)}, "
        f"dev={sum(len(r['extractions']) for r in dev_rows)}, "
        f"test={sum(len(r['extractions']) for r in test_rows)}"
    )

    # Diagnostic only: never alters the official split.
    train_sentences = {r["source"] for r in train_rows}
    dev_sentences = {r["source"] for r in dev_rows}
    test_sentences = {r["source"] for r in test_rows}
    print(
        "Exact-sentence overlap diagnostic: "
        f"train/dev={len(train_sentences & dev_sentences)}, "
        f"train/test={len(train_sentences & test_sentences)}, "
        f"dev/test={len(dev_sentences & test_sentences)}"
    )

    return train_rows, dev_rows, test_rows


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def make_optimizer_and_scheduler(
    model: torch.nn.Module,
    num_batches: int,
    epochs: int,
    grad_accum: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-8,
    )

    updates_per_epoch = math.ceil(num_batches / grad_accum)
    total_steps = max(1, updates_per_epoch * epochs)
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


@torch.no_grad()
def mean_loss(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    for batch in loader:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
            if key in {"input_ids", "attention_mask", "labels"}
        }

        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16
            else contextlib.nullcontext()
        )

        with amp_context:
            loss = model(**batch).loss

        total += float(loss.detach().float().item())
        count += 1

    return total / max(count, 1)


def train_loop(
    model,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    device: torch.device,
    epochs: int,
    grad_accum: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    max_grad_norm: float,
) -> None:
    optimizer, scheduler = make_optimizer_and_scheduler(
        model=model,
        num_batches=len(train_loader),
        epochs=epochs,
        grad_accum=grad_accum,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
    )

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")

        for step, batch in enumerate(progress, start=1):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask", "labels"}
            }

            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else contextlib.nullcontext()
            )

            with amp_context:
                loss = model(**batch).loss
                scaled_loss = loss / grad_accum

            scaled_loss.backward()
            running_loss += float(loss.detach().float().item())

            if step % grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(loss=f"{loss.detach().float().item():.4f}")

        train_mean = running_loss / max(len(train_loader), 1)
        dev_mean = mean_loss(model, dev_loader, device)

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_mean:.6f} dev_loss={dev_mean:.6f}"
        )


# ---------------------------------------------------------------------------
# GPT-2: causal generation
# ---------------------------------------------------------------------------

def gpt_prompt(sentence: str) -> str:
    return f"Sentence: {sentence}\nOpenIE:"


class GPT2OIEDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]

        prompt_ids = self.tokenizer(
            gpt_prompt(row["source"]),
            add_special_tokens=False,
        )["input_ids"]

        target_ids = self.tokenizer(
            " " + row["target"],
            add_special_tokens=False,
        )["input_ids"]

        if self.tokenizer.eos_token_id is not None:
            target_ids.append(self.tokenizer.eos_token_id)

        if self.max_length < 2:
            raise ValueError("GPT-2 --max-seq-length must be at least 2")

        # Preserve target capacity by truncating the prompt first.
        prompt_ids = prompt_ids[: self.max_length - 1]
        remaining = self.max_length - len(prompt_ids)
        target_ids = target_ids[:remaining]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids.copy()
        attention_mask = [1] * len(input_ids)

        pad_len = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def sequence_confidences(generated, batch_size: int) -> list[float]:
    """
    Beam-search sequence scores are used only to rank predictions for the
    benchmark precision-recall curve.  If unavailable, all sequences receive
    equal confidence.
    """
    scores = getattr(generated, "sequences_scores", None)
    if scores is None:
        return [1.0] * batch_size
    return [float(score.detach().cpu().item()) for score in scores]


@torch.no_grad()
def generate_gpt2(
    model,
    tokenizer,
    rows,
    device,
    batch_size: int,
    max_seq_length: int,
    max_new_tokens: int,
    num_beams: int,
):
    model.eval()

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    max_positions = (
        getattr(model.config, "n_positions", None)
        or getattr(model.config, "max_position_embeddings", None)
        or 1024
    )

    max_source_length = min(
        max_seq_length,
        max(1, int(max_positions) - max_new_tokens),
    )

    predictions: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(rows), batch_size), desc="GPT-2 test"):
        chunk = rows[start : start + batch_size]
        prompts = [gpt_prompt(row["source"]) for row in chunk]

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_length,
            add_special_tokens=False,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_width = encoded["input_ids"].shape[1]

        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        continuation_ids = generated.sequences[:, input_width:]
        decoded = tokenizer.batch_decode(
            continuation_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        confidences = sequence_confidences(generated, len(chunk))

        for row, output, confidence in zip(chunk, decoded, confidences):
            predictions.append(
                {
                    "text": row["source"],
                    "gold_target": row["target"],
                    "output": output.strip(),
                    "confidence": confidence,
                }
            )

    tokenizer.padding_side = old_padding_side
    return predictions


def run_gpt2(args, model_path, train_rows, dev_rows, test_rows, device):
    tokenizer_name = args.tokenizer or "gpt2-medium"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    if tokenizer.eos_token_id is None:
        raise ValueError("GPT-2 tokenizer must provide an EOS token.")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    model.to(device)

    train_ds = GPT2OIEDataset(train_rows, tokenizer, args.max_seq_length)
    dev_ds = GPT2OIEDataset(dev_rows, tokenizer, args.max_seq_length)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )

    train_loop(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        device=device,
        epochs=args.epochs,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
    )

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    return generate_gpt2(
        model=model,
        tokenizer=tokenizer,
        rows=test_rows,
        device=device,
        batch_size=args.eval_batch_size,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )


# ---------------------------------------------------------------------------
# BERT: warm-started BERT2BERT generation
# ---------------------------------------------------------------------------

class Bert2BertOIEDataset(Dataset):
    def __init__(
        self,
        rows,
        tokenizer,
        max_source_length: int,
        max_target_length: int,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]

        source = self.tokenizer(
            row["source"],
            truncation=True,
            padding="max_length",
            max_length=self.max_source_length,
            return_attention_mask=True,
        )

        # EncoderDecoderModel shifts labels right and inserts decoder_start_token_id
        # itself.  Therefore labels should NOT begin with an additional [CLS].
        target_ids = self.tokenizer(
            row["target"],
            add_special_tokens=False,
            truncation=True,
            max_length=max(1, self.max_target_length - 1),
        )["input_ids"]

        target_ids.append(self.tokenizer.sep_token_id)
        target_ids = target_ids[: self.max_target_length]

        pad_len = self.max_target_length - len(target_ids)
        labels = target_ids + [-100] * pad_len

        return {
            "input_ids": torch.tensor(source["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                source["attention_mask"],
                dtype=torch.long,
            ),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def configure_bert2bert(model, tokenizer) -> None:
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("BERT tokenizer must provide CLS and SEP tokens.")
    if tokenizer.pad_token_id is None:
        raise ValueError("BERT tokenizer must provide a PAD token.")

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.decoder_start_token_id = tokenizer.cls_token_id
        model.generation_config.eos_token_id = tokenizer.sep_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id


@torch.no_grad()
def generate_bert2bert(
    model,
    tokenizer,
    rows,
    device,
    batch_size: int,
    max_source_length: int,
    max_new_tokens: int,
    num_beams: int,
):
    model.eval()
    predictions: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(rows), batch_size), desc="BERT2BERT test"):
        chunk = rows[start : start + batch_size]

        encoded = tokenizer(
            [row["source"] for row in chunk],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            decoder_start_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        decoded = tokenizer.batch_decode(
            generated.sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        confidences = sequence_confidences(generated, len(chunk))

        for row, output, confidence in zip(chunk, decoded, confidences):
            predictions.append(
                {
                    "text": row["source"],
                    "gold_target": row["target"],
                    "output": output.strip(),
                    "confidence": confidence,
                }
            )

    return predictions


def run_bert(args, model_path, train_rows, dev_rows, test_rows, device):
    tokenizer_name = args.tokenizer or "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    if tokenizer.pad_token_id is None:
        raise ValueError("BERT tokenizer must provide a PAD token.")

    print(f"Initializing BERT2BERT encoder and decoder from: {model_path}")
    print(
        "Decoder cross-attention is newly initialized for both baseline BERT "
        "and PR-BERT."
    )

    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        model_path,
        model_path,
    )
    configure_bert2bert(model, tokenizer)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if hasattr(model.config.decoder, "use_cache"):
        model.config.decoder.use_cache = False

    model.to(device)

    train_ds = Bert2BertOIEDataset(
        train_rows,
        tokenizer,
        max_source_length=args.max_seq_length,
        max_target_length=args.max_target_length,
    )
    dev_ds = Bert2BertOIEDataset(
        dev_rows,
        tokenizer,
        max_source_length=args.max_seq_length,
        max_target_length=args.max_target_length,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )

    train_loop(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        device=device,
        epochs=args.epochs,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
    )

    if hasattr(model.config.decoder, "use_cache"):
        model.config.decoder.use_cache = True

    return generate_bert2bert(
        model=model,
        tokenizer=tokenizer,
        rows=test_rows,
        device=device,
        batch_size=args.eval_batch_size,
        max_source_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )


# ---------------------------------------------------------------------------
# Generated-string parsing and case restoration
# ---------------------------------------------------------------------------

def split_extraction_chunks(output: str) -> list[str]:
    """
    Primary separator is ' || '.  A light fallback also separates adjacent
    parenthesized tuples if the model omits the bars.
    """
    output = output.strip()
    if not output:
        return []

    chunks = re.split(r"\s*\|\|\s*", output)

    if len(chunks) == 1:
        # Common malformed-but-recoverable output: "(...) (...)"
        adjacent = re.findall(r"\([^)]*\)", output)
        if len(adjacent) > 1:
            chunks = adjacent

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def parse_generated_extractions(output: str) -> list[dict[str, Any]]:
    """
    Parse:
        (ARG1; REL; ARG2; ARG3) || (ARG1; REL; ARG2)

    At least ARG1 and REL are required.  Malformed tuples are ignored rather
    than crashing the entire benchmark run.
    """
    parsed: list[dict[str, Any]] = []

    for chunk in split_extraction_chunks(output):
        inner = chunk.strip()
        if inner.startswith("("):
            inner = inner[1:]
        if inner.endswith(")"):
            inner = inner[:-1]

        fields = [field.strip() for field in inner.split(";")]
        fields = [field for field in fields if field]

        if len(fields) < 2:
            continue

        arg1 = fields[0]
        predicate = fields[1]
        remaining_args = fields[2:]

        if not arg1 or not predicate:
            continue

        parsed.append(
            {
                "predicate": predicate,
                "args": [arg1] + remaining_args,
            }
        )

    return parsed


def restore_field_case(field: str, sentence: str) -> str:
    """
    bert-base-uncased necessarily decodes lowercase text.  The official
    lexical matcher is called with ignoreCase=True, but the historical
    implementation does not actually lowercase inside lexicalMatch.

    If a generated field exactly matches a contiguous source substring
    ignoring case (allowing flexible whitespace), restore the source casing.
    This changes casing only; it never substitutes different words.
    """
    field = field.strip()
    if not field:
        return field

    tokens = field.split()
    if not tokens:
        return field

    pattern = r"\s+".join(re.escape(token) for token in tokens)
    match = re.search(pattern, sentence, flags=re.IGNORECASE)
    return match.group(0) if match else field


def restore_extraction_case(extraction: dict[str, Any], sentence: str):
    return {
        "predicate": restore_field_case(extraction["predicate"], sentence),
        "args": [
            restore_field_case(arg, sentence)
            for arg in extraction["args"]
        ],
    }


# ---------------------------------------------------------------------------
# Official oie-benchmark integration
# ---------------------------------------------------------------------------

def validate_benchmark_repo(repo_dir: str | Path) -> Path:
    repo_dir = Path(repo_dir).resolve()
    required = [
        repo_dir / "benchmark.py",
        repo_dir / "matcher.py",
        repo_dir / "oie_readers",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"--benchmark-repo is not a valid oie-benchmark clone: {repo_dir}\n"
            f"Missing: {missing}"
        )
    return repo_dir


def prepare_old_benchmark_compatibility(repo_dir: str | Path):
    """
    oie-benchmark was released against scikit-learn 0.18 and imports
    sklearn.preprocessing.data.binarize, a path removed in modern sklearn.

    Install a small module-path compatibility shim, ensure NLTK resources used
    by matcher.py are available, then import the cloned repo's evaluator.
    """
    repo_dir = validate_benchmark_repo(repo_dir)

    import sklearn.preprocessing

    if "sklearn.preprocessing.data" not in sys.modules:
        shim = types.ModuleType("sklearn.preprocessing.data")
        shim.binarize = sklearn.preprocessing.binarize
        sys.modules["sklearn.preprocessing.data"] = shim

    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    try:
        stopwords.words("english")
    except LookupError:
        print("NLTK stopwords not found; attempting download...")
        nltk.download("stopwords", quiet=True)
        try:
            stopwords.words("english")
        except LookupError as exc:
            raise RuntimeError(
                "NLTK stopwords are required by oie-benchmark/matcher.py. "
                "Install them once with: python -m nltk.downloader stopwords"
            ) from exc

    try:
        WordNetLemmatizer().lemmatize("tests")
    except LookupError:
        print("NLTK wordnet not found; attempting download...")
        nltk.download("wordnet", quiet=True)
        try:
            WordNetLemmatizer().lemmatize("tests")
        except LookupError as exc:
            raise RuntimeError(
                "NLTK wordnet is required by oie-benchmark/Matcher.lexicalMatch. "
                "Install it once with: python -m nltk.downloader wordnet"
            ) from exc

    repo_string = str(repo_dir)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)

    from benchmark import Benchmark
    from matcher import Matcher
    from oie_readers.extraction import Extraction

    return Benchmark, Matcher, Extraction


def predictions_to_official_objects(predictions, Extraction):
    predicted = defaultdict(list)
    parsed_count = 0

    for row in predictions:
        sentence = row["text"]
        confidence = float(row["confidence"])

        parsed = parse_generated_extractions(row["output"])
        for extraction_data in parsed:
            extraction_data = restore_extraction_case(
                extraction_data,
                sentence,
            )

            extraction = Extraction(
                pred=extraction_data["predicate"],
                sent=sentence,
                confidence=confidence,
            )

            for arg in extraction_data["args"]:
                extraction.addArg(arg)

            predicted[sentence].append(extraction)
            parsed_count += 1

    return predicted, parsed_count


def read_pr_curve(path: str | Path):
    precision: list[float] = []
    recall: list[float] = []

    with open(path, encoding="utf-8") as f:
        next(f, None)  # Precision\tRecall header
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2:
                continue
            precision.append(float(fields[0]))
            recall.append(float(fields[1]))

    return np.asarray(precision, dtype=float), np.asarray(recall, dtype=float)


def summarize_pr_curve(precision: np.ndarray, recall: np.ndarray):
    if len(precision) == 0:
        return {
            "best_precision": 0.0,
            "best_recall": 0.0,
            "best_f1": 0.0,
            "pr_auc": 0.0,
        }

    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.argmax(f1))

    order = np.argsort(recall)
    recall_sorted = recall[order]
    precision_sorted = precision[order]

    pr_auc = (
        float(auc(recall_sorted, precision_sorted))
        if len(precision_sorted) > 1
        else 0.0
    )

    return {
        "best_precision": float(precision[best_index]),
        "best_recall": float(recall[best_index]),
        "best_f1": float(f1[best_index]),
        "pr_auc": pr_auc,
    }


def official_evaluate(
    predictions,
    gold_path: str | Path,
    benchmark_repo: str | Path,
    pr_curve_path: str | Path,
):
    Benchmark, Matcher, Extraction = prepare_old_benchmark_compatibility(
        benchmark_repo
    )

    predicted, parsed_count = predictions_to_official_objects(
        predictions,
        Extraction,
    )

    pr_curve_path = Path(pr_curve_path)
    pr_curve_path.parent.mkdir(parents=True, exist_ok=True)

    if parsed_count == 0:
        # Benchmark.compare eventually calls sklearn precision_recall_curve,
        # which cannot construct a meaningful curve from zero predictions.
        with open(pr_curve_path, "w", encoding="utf-8") as f:
            f.write("Precision\tRecall\n")
            f.write("0.0\t0.0\n")
        metrics = summarize_pr_curve(
            np.asarray([0.0]),
            np.asarray([0.0]),
        )
        metrics["num_parsed_predictions"] = 0
        return metrics

    benchmark = Benchmark(str(gold_path))
    benchmark.compare(
        predicted=predicted,
        matchingFunc=Matcher.lexicalMatch,
        output_fn=str(pr_curve_path),
    )

    precision, recall = read_pr_curve(pr_curve_path)
    metrics = summarize_pr_curve(precision, recall)
    metrics["num_parsed_predictions"] = parsed_count
    return metrics


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def write_results(
    args,
    run_name: str,
    resolved_model_path: str,
    predictions,
    metrics,
    pr_curve_path: Path,
):
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = results_dir / f"{run_name}.predictions.jsonl"
    with open(prediction_path, "w", encoding="utf-8") as f:
        for row in predictions:
            record = dict(row)
            record["parsed_extractions"] = parse_generated_extractions(
                row["output"]
            )
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    metrics_record = dict(metrics)
    metrics_record.update(
        {
            "run_name": run_name,
            "architecture": args.arch,
            "model_source": args.model,
            "resolved_model_path": resolved_model_path,
            "tokenizer": args.tokenizer,
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "grad_accum": args.grad_accum,
            "gold_test_file": str(Path(args.data_dir) / "test.oie"),
            "benchmark_repo": str(Path(args.benchmark_repo)),
            "matching_function": "Matcher.lexicalMatch",
            "predictions_file": str(prediction_path),
            "pr_curve_file": str(pr_curve_path),
        }
    )

    metrics_path = results_dir / f"{run_name}.metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_record, indent=2),
        encoding="utf-8",
    )

    return prediction_path, metrics_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune GPT-2/GPT-2-PR or BERT/BERT-PR on official OIE2016 "
            "train/dev splits and evaluate on official test with oie-benchmark."
        )
    )

    parser.add_argument("arch", choices=["bert", "gpt2"])
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Hugging Face model id or local Hugging Face checkpoint directory.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Tokenizer name/path. Defaults to bert-base-uncased for BERT and "
            "gpt2-medium for GPT-2."
        ),
    )
    parser.add_argument("--run-name", default=None)

    parser.add_argument(
        "--benchmark-repo",
        default=None,
        help=(
            "Path to the cloned gabrielStanovsky/oie-benchmark repository. "
            "Auto-detected as ./oie-benchmark or the current directory."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Directory containing official train.oie, dev.oie, test.oie. "
            "Defaults to <benchmark-repo>/oie_corpus."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default="outputs/oie_results",
    )

    parser.add_argument("-e", "--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Default effective train batch size: 4 x 8 = 32.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Default: 5e-5 for GPT-2; 3e-5 for BERT2BERT.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action="store_true")

    return parser.parse_args()


def resolve_repo_layout(args) -> None:
    """Resolve the cloned benchmark and official split directories."""
    if args.benchmark_repo is None:
        if Path("oie-benchmark/benchmark.py").is_file():
            args.benchmark_repo = "oie-benchmark"
        elif Path("benchmark.py").is_file() and Path("oie_readers").is_dir():
            args.benchmark_repo = "."
        else:
            raise FileNotFoundError(
                "Could not auto-detect oie-benchmark. Expected either "
                "./oie-benchmark/benchmark.py or ./benchmark.py. "
                "Pass --benchmark-repo explicitly if it is elsewhere."
            )

    if args.data_dir is None:
        args.data_dir = str(Path(args.benchmark_repo) / "oie_corpus")


def validate_args(args) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.train_batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.grad_accum <= 0:
        raise ValueError("--grad-accum must be positive")
    if not (0.0 <= args.warmup_ratio < 1.0):
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if args.max_seq_length <= 1:
        raise ValueError("--max-seq-length must be greater than 1")
    if args.max_target_length <= 1:
        raise ValueError("--max-target-length must be greater than 1")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.num_beams <= 0:
        raise ValueError("--num-beams must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")


def main() -> None:
    args = parse_args()
    resolve_repo_layout(args)
    validate_args(args)

    if args.learning_rate is None:
        args.learning_rate = 5e-5 if args.arch == "gpt2" else 3e-5

    set_seed(args.seed)

    # Validate data AND import the official evaluator before loading/training a
    # large model. This prevents discovering an old-repo dependency problem
    # only after a multi-hour fine-tuning run.
    validate_benchmark_repo(args.benchmark_repo)
    train_rows, dev_rows, test_rows = load_official_splits(args.data_dir)
    prepare_old_benchmark_compatibility(args.benchmark_repo)
    print("Official oie-benchmark evaluator preflight: PASS")

    model_path = resolve_model_path(args.model)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer_display = args.tokenizer or (
        "bert-base-uncased" if args.arch == "bert" else "gpt2-medium"
    )

    print("\n============================================================")
    print(f"Architecture:       {args.arch}")
    print(f"Model source:       {args.model}")
    print(f"Resolved model:     {model_path}")
    print(f"Tokenizer:          {tokenizer_display}")
    print(f"Seed:               {args.seed}")
    print(f"Epochs:             {args.epochs}")
    print(f"Learning rate:      {args.learning_rate}")
    print(f"Device:             {device}")
    print(f"Train:              {Path(args.data_dir) / 'train.oie'}")
    print(f"Dev:                {Path(args.data_dir) / 'dev.oie'}")
    print(f"Test:               {Path(args.data_dir) / 'test.oie'}")
    print("Evaluation:         Benchmark + Matcher.lexicalMatch")
    print("Checkpoint saving:  disabled")
    print("============================================================\n")

    if args.arch == "gpt2":
        predictions = run_gpt2(
            args,
            model_path,
            train_rows,
            dev_rows,
            test_rows,
            device,
        )
    else:
        predictions = run_bert(
            args,
            model_path,
            train_rows,
            dev_rows,
            test_rows,
            device,
        )

    run_name = args.run_name or (
        f"oie-{args.arch}-{safe_name(args.model)}-"
        f"{args.epochs}ep-seed{args.seed}"
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    pr_curve_path = results_dir / f"{run_name}.pr.dat"

    metrics = official_evaluate(
        predictions=predictions,
        gold_path=Path(args.data_dir) / "test.oie",
        benchmark_repo=args.benchmark_repo,
        pr_curve_path=pr_curve_path,
    )

    prediction_path, metrics_path = write_results(
        args=args,
        run_name=run_name,
        resolved_model_path=model_path,
        predictions=predictions,
        metrics=metrics,
        pr_curve_path=pr_curve_path,
    )

    print("\n================ OIE2016 OFFICIAL TEST =================")
    print(f"Run:             {run_name}")
    print(f"Best Precision:  {metrics['best_precision']:.6f}")
    print(f"Best Recall:     {metrics['best_recall']:.6f}")
    print(f"Best F1:         {metrics['best_f1']:.6f}")
    print(f"PR AUC:          {metrics['pr_auc']:.6f}")
    print(f"Parsed outputs:  {metrics['num_parsed_predictions']}")
    print(f"Predictions:     {prediction_path}")
    print(f"PR curve:        {pr_curve_path}")
    print(f"Metrics:         {metrics_path}")
    print("========================================================\n")


if __name__ == "__main__":
    main()