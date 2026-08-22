import os
import glob
import argparse
import random
import numpy as np
import re

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    Trainer,
    set_seed,
)

# CaRB imports (ensure CaRB repo is in PYTHONPATH)
from carb import Benchmark
from matcher import Matcher
from oie_readers.tabReader import TabReader


def seed_everything(seed: int):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Model loader
# ----------------------------
def load_causal_model_and_tokenizer(model_name: str):
    """
    Supports:
      - GPT2 / causal LMs via AutoModelForCausalLM
      - BERT checkpoints by forcing decoder mode (BertLMHeadModel)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        # GPT2 has no pad; use eos as pad
        tokenizer.pad_token = tokenizer.eos_token

    if "bert" in model_name.lower():
        # Force BERT into decoder / causal mode so generate() works
        from transformers import BertLMHeadModel

        cfg = AutoConfig.from_pretrained(model_name)
        cfg.is_decoder = True
        cfg.add_cross_attention = False

        tokenizer.eos_token = tokenizer.sep_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.sep_token

        model = BertLMHeadModel.from_pretrained(model_name, config=cfg)
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


# ----------------------------
# OIE data loading (your folder)
# ----------------------------
def read_oie_gold_file(path: str):
    """
    Gold file format (tabs): SENT \t REL \t ARG1 \t ARG2 ...
    Returns dict: sentence -> list of (rel, [arg1,arg2,...])
    """
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                # need at least sent, rel, arg1
                continue
            sent = parts[0].strip()
            rel = parts[1].strip()
            args = [p.strip() for p in parts[2:] if p.strip()]
            d.setdefault(sent, []).append((rel, args))
    return d


def load_oie2016_splits(oie_dir: str, seed: int):
    """
    If files named train*.oie/dev*.oie/test*.oie exist, use them.
    Else: merge all *.oie and do a 90/10 split; use dev as test.
    """
    paths = sorted(glob.glob(os.path.join(oie_dir, "*.oie")))
    if not paths:
        raise FileNotFoundError(f"No .oie files found in: {oie_dir}")

    lower = {os.path.basename(p).lower(): p for p in paths}

    def pick(prefixes):
        for name, p in lower.items():
            if any(name.startswith(pref) for pref in prefixes):
                return p
        return None

    train_p = pick(["train"])
    dev_p = pick(["dev", "valid", "val"])
    test_p = pick(["test"])

    if train_p and dev_p and test_p:
        return read_oie_gold_file(train_p), read_oie_gold_file(dev_p), read_oie_gold_file(test_p)

    merged = {}
    for p in paths:
        cur = read_oie_gold_file(p)
        for s, triples in cur.items():
            merged.setdefault(s, []).extend(triples)

    sents = list(merged.keys())
    rng = random.Random(seed)
    rng.shuffle(sents)

    n = len(sents)
    n_train = max(1, int(0.9 * n))
    train_sents = sents[:n_train]
    dev_sents = sents[n_train:]

    train_map = {s: merged[s] for s in train_sents}
    dev_map = {s: merged[s] for s in dev_sents}
    test_map = dev_map  # simple default

    return train_map, dev_map, test_map


# ----------------------------
# Prompting format
# ----------------------------
OIE_SEP = "|||"


def build_oie_prompt(sentence: str) -> str:
    return (
        f"Sentence: {sentence}\n"
        f"Extractions (one per line as: ARG1 {OIE_SEP} REL {OIE_SEP} ARG2 ...):\n"
    )


def gold_to_target(triples):
    """
    triples: list of (rel, [arg1,arg2,...])
    Produce lines: ARG1 ||| REL ||| ARG2 ...
    """
    lines = []
    for rel, args in triples:
        if not args:
            continue
        # args already includes arg1,arg2,...
        fields = [args[0], rel] + args[1:]
        lines.append(f" {OIE_SEP} ".join(fields))
    return "\n".join(lines)


def parse_generated_extractions(decoded: str):
    """
    Parse lines after "Extractions" into list of (rel, [arg1,arg2,...]).
    """
    if "Extractions" in decoded:
        decoded = decoded.split("Extractions", 1)[1]
        if ":\n" in decoded:
            decoded = decoded.split(":\n", 1)[1]
        decoded = decoded.strip()

    out = []
    for line in decoded.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in re.split(r"\s*\|\s*\|\s*\|\s*", line)]
        if len(parts) < 3:
            continue
        arg1 = parts[0]
        rel = parts[1]
        other_args = parts[2:]
        args = [arg1] + other_args
        out.append((rel, args))
    return out


def preprocess_oie(example, tokenizer, max_length=512):
    prompt = build_oie_prompt(example["sentence"])
    gold = example["target"]
    full_text = prompt + gold

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

    # mask padding
    attn = tokenized["attention_mask"]
    labels = [lab if m == 1 else -100 for lab, m in zip(labels, attn)]

    tokenized["labels"] = labels
    return tokenized


# ----------------------------
# Write predictions in TabReader format
# ----------------------------
def write_tab_predictions(out_path: str, sentences, all_triples):
    """
    Each line: SENT \t CONF \t REL \t ARG1 \t ARG2 ...
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for sent, triples in zip(sentences, all_triples):
            for rel, args in triples:
                if not args:
                    continue
                conf = "1.0"  # simple single operating point
                row = [sent, conf, rel] + args
                f.write("\t".join(row) + "\n")


# ----------------------------
# CaRB evaluation (return p,r,f1)
# ----------------------------
def evaluate_carb_prf(gold_oie_path: str, pred_tab_path: str, matching="binary_lenient"):
    tr = TabReader()
    tr.read(pred_tab_path)

    b = Benchmark(gold_oie_path)

    if matching == "binary_strict":
        mfunc = Matcher.binary_tuple_match
    elif matching == "simple":
        mfunc = Matcher.simple_tuple_match
    elif matching == "exact":
        mfunc = Matcher.argMatch
    elif matching == "pred":
        mfunc = Matcher.predMatch
    elif matching == "lexical":
        mfunc = Matcher.lexicalMatch
    elif matching == "strict":
        mfunc = Matcher.tuple_match
    else:
        # CaRB default, spelling error where it is spelt linient
        mfunc = Matcher.binary_linient_tuple_match

    auc, optimal = b.compare(predicted=tr.oie, matchingFunc=mfunc, output_fn=os.devnull)
    # optimal = (p, r, f1)
    p, r, f1 = float(optimal[0]), float(optimal[1]), float(optimal[2])
    return p, r, f1, float(auc)


def make_gold_all_file(oie_dir: str, out_path: str):
    """
    CaRB expects a single gold file path. We just concatenate all *.oie.
    """
    if os.path.exists(out_path):
        return
    paths = sorted(glob.glob(os.path.join(oie_dir, "*.oie")))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for p in paths:
            with open(p, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if line:
                        out.write(line + "\n")


# ----------------------------
# Main task: conll16 / OIE2016
# ----------------------------
def run_conll2016_oie(args, seed):
    seed_everything(seed)
    print(f"\n===== Training {args.model} on CoNLL-2016 / OIE2016 (CaRB) =====")

    oie_dir = os.path.join("OIE", "oie_corpus")
    train_map, dev_map, test_map = load_oie2016_splits(oie_dir, seed=seed)

    # Build HF datasets (sentence, target)
    train_rows = [{"sentence": s, "target": gold_to_target(triples)} for s, triples in train_map.items()]
    dev_rows = [{"sentence": s, "target": gold_to_target(triples)} for s, triples in dev_map.items()]

    # test small subset of data first
    # TOREMOVE
    # train_rows = train_rows[:100]
    # dev_rows = dev_rows[:100]

    train_raw = Dataset.from_list(train_rows)
    dev_raw = Dataset.from_list(dev_rows)

    model, tokenizer = load_causal_model_and_tokenizer(args.model)

    max_len = args.max_length

    train_ds = train_raw.map(
        lambda x: preprocess_oie(x, tokenizer, max_length=max_len),
        remove_columns=train_raw.column_names,
    )
    dev_ds = dev_raw.map(
        lambda x: preprocess_oie(x, tokenizer, max_length=max_len),
        remove_columns=dev_raw.column_names,
    )

    save_dir = os.path.join("checkpoints", f"{args.model.replace('/', '_')}_conll16_oie_e{args.epochs}")
    os.makedirs(save_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=save_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        seed=seed,
        data_seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
    )

    trainer.train()

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Generate predictions on test sentences
    print("\n===== CoNLL-2016 OIE Evaluation (CaRB) =====")
    model.eval()
    device = next(model.parameters()).device
    print("GEN DEVICE:", next(model.parameters()).device)

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.sep_token_id

    test_sents = list(test_map.keys())

    # debugging empty bert output
    if args.debug_n and args.debug_n > 0:
        test_sents = test_sents[:args.debug_n]

    # SMALL TEST, TO REMOVE
    # test_sents = test_sents[:100]

    pred_all = []

    for sent in test_sents:
        prompt = build_oie_prompt(sent)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        ).to(device)

        # with torch.no_grad():
        #     gen_ids = model.generate(
        #         **inputs,
        #         do_sample=False,
        #         num_beams=1,
        #         max_new_tokens=args.gen_max_new_tokens,
        #         eos_token_id=eos_id,
        #         pad_token_id=tokenizer.pad_token_id,
        #     )

        #debugging empty bert output
        gen_kwargs = dict(
            do_sample=False,
            num_beams=1,
            max_new_tokens=args.gen_max_new_tokens,
            eos_token_id=eos_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        # For BERT-as-decoder, force it to produce something
        if "bert" in args.model.lower():
            gen_kwargs.update(
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
                min_new_tokens=32,
                bad_words_ids=[[eos_id]],  # avoid immediately generating [SEP]
            )

        with torch.no_grad():
            gen_ids = model.generate(**inputs, **gen_kwargs)



        decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        if len(pred_all) == 0:
            print("PROMPT:\n", prompt)
            print("DECODED:\n", decoded)
            print("GOLD:\n", gold_to_target(test_map[sent]))

        pred_triples = parse_generated_extractions(decoded)
        pred_all.append(pred_triples)

    pred_path = os.path.join("outputs", "generated", f"{args.model.replace('/', '_')}_conll16_preds.tab")
    write_tab_predictions(pred_path, test_sents, pred_all)

    gold_all_path = os.path.join("OIE", "oie_corpus", "all.oie")
    make_gold_all_file(oie_dir, gold_all_path)

    p, r, f1, auc = evaluate_carb_prf(
        gold_oie_path=gold_all_path,
        pred_tab_path=pred_path,
        matching=args.oie_matching,
    )

    # Print like your other tasks (plus AUC)
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
    print(f"CaRB AUC: {auc:.4f}")
    print("\ncompleted conll16 (oie)\n")

    # IMPORTANT: return (p,r,f1) exactly like your other run_* functions
    return p, r, f1


# ----------------------------
# Results CSV
# ----------------------------
def write_results_csv(rows, out_path):
    import csv

    fieldnames = ["model", "task", "run", "seed", "precision", "recall", "f1"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="model to fine-tune (gpt2, checkpoints/gpt2_yelp_pr, bert-base-uncased, felflare/bert-restore-punctuation)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="conll16",
        help="task to fine-tune (conll16 for OIE2016/CaRB in this file)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)

    # debugging empty bert output
    parser.add_argument("--debug_n", type=int, default=0, help="use only N examples for train/dev/test (0=full)")

    # OIE-specific knobs
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--gen_max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--oie_matching",
        type=str,
        default="binary_lenient",
        help="CaRB matching: binary_lenient (default), binary_strict, simple, exact, pred, lexical, strict",
    )

    args = parser.parse_args()

    results = []
    for i in range(10):
        if args.task == "conll16":
            p, r, f1 = run_conll2016_oie(args, i)
        else:
            p, r, f1 = 0.0, 0.0, 0.0
            print("task not available in this file (only conll16 is implemented here)")

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

    save_csv = os.path.join("outputs", "generated", f"{args.model.replace('/', '_')}_{args.task}.csv")
    write_results_csv(results, save_csv)
    print(f"\nSaved results to {save_csv}\n")