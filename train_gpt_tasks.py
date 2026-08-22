import os
import argparse
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
)
import torch
import random
import string

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import csv


def seed_everything(seed: int):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_tag(tok: str):
    # strip whitespace and punctuation
    return (tok or "").strip().strip(string.punctuation)

def sanitize_to_label_set(pred_tags, label_set):
    # map unknown tokens to "O" and clean up whitespace/punctuation
    cleaned = []
    for t in pred_tags:
        t = clean_tag(t)
        cleaned.append(t if t in label_set else "O")
    return cleaned

def seqeval_prf1(gold_seqs, pred_seqs):
    # Call default seqeval metrics
    p = precision_score(gold_seqs, pred_seqs)
    r = recall_score(gold_seqs, pred_seqs)
    f1 = f1_score(gold_seqs, pred_seqs)
    return p, r, f1


# Note that conll and genia use the same evaluation
def evaluate_conll_ner(model, dataset, tokenizer, label_names, max_length=256, gen_max_new_tokens=256):
    model.eval()
    device = next(model.parameters()).device
    label_set = set(label_names)

    gold_seqs, pred_seqs = [], []

    for example in dataset:
        tokens = example["tokens"]
        gold_tags = [label_names[i] for i in example["ner_tags"]]

        prompt = build_ner_prompt(tokens)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=min(gen_max_new_tokens, len(gold_tags) + 20),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        # Prefer strict delimiter parsing; if missing, treat as empty (=> all O after padding)
        if "NER tags:" in decoded:
            pred_part = decoded.split("NER tags:", 1)[1].strip()
            pred_tags = pred_part.split()
        else:
            pred_tags = []

        pred_tags = sanitize_to_label_set(pred_tags, label_set)

        # pad/truncate to gold length
        if len(pred_tags) < len(gold_tags):
            pred_tags += ["O"] * (len(gold_tags) - len(pred_tags))
        pred_tags = pred_tags[:len(gold_tags)]

        gold_seqs.append(gold_tags)
        pred_seqs.append(pred_tags)

    return seqeval_prf1(gold_seqs, pred_seqs)


def evaluate_ontonotes_ner(model, dataset, tokenizer, id2label, max_length=256, gen_max_new_tokens=256):
    model.eval()
    device = next(model.parameters()).device
    label_set = set(id2label.values())

    gold_seqs, pred_seqs = [], []

    for example in dataset:
        tokens = example["tokens"]
        gold_tags = [id2label[i] for i in example["tags"]]

        prompt = build_ner_prompt(tokens)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=min(gen_max_new_tokens, len(gold_tags) + 20),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        if "NER tags:" in decoded:
            pred_part = decoded.split("NER tags:", 1)[1].strip()
            pred_tags = pred_part.split()
        else:
            pred_tags = []

        pred_tags = sanitize_to_label_set(pred_tags, label_set)

        if len(pred_tags) < len(gold_tags):
            pred_tags += ["O"] * (len(gold_tags) - len(pred_tags))
        pred_tags = pred_tags[:len(gold_tags)]

        gold_seqs.append(gold_tags)
        pred_seqs.append(pred_tags)

    return seqeval_prf1(gold_seqs, pred_seqs)



#from bert testing
# used for POS 
def prf1(num_correct: int, num_attempted: int, num_gold: int) -> tuple[float, float, float]:
    precision = num_correct / num_attempted if num_attempted else 0.0
    recall = num_correct / num_gold if num_gold else 0.0
    
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def list_hamming_dist(a, b):
    """Count number of index matches"""
    return sum([1 if a[i] == b[i] else 0 for i in range(min(len(a), len(b)))])

def score(texts: list[str], outputs: list[str], targets: list[str], strict = False) -> tuple[float, float, float]:
    """Score POS by matching"""
    num_correct, num_attempted, num_gold = 0, 0, 0
    for text, output, target in zip(texts, outputs, targets):
        output_tags, target_tags = output.split(), target.split()
        num_attempted += len(output_tags)
        num_gold += len(target_tags)
        num_correct += list_hamming_dist(output_tags, target_tags)
    return prf1(num_correct, num_attempted, num_gold)
#############################################

#gpt2 prompt for POS (conll00)
def build_pos_prompt(tokens):
    sent = " ".join(tokens)
    return f"Sentence: {sent}\nPOS tags:"

def build_ner_prompt(tokens):
    sent = " ".join(tokens)
    return f"Sentence: {sent}\nNER tags:"

#conll12 srl
def build_srl_prompt(tokens, predicate_index):
    # Put [PRED] markers in the sentence so the model knows which predicate.
    # Gold tags should still be one-per-original-token (exclude markers).
    toks = tokens.copy()
    pred = predicate_index

    marked = toks[:pred] + ["[PRED]", toks[pred], "[/PRED]"] + toks[pred + 1 :]
    sent = " ".join(marked)

    # keep the delimiter consistent with eval parsing
    return f"Sentence: {sent}\nSRL tags:"



# helpers for SRL (conll12)
# same as for training bert
def bio_to_spans(tags):
    spans = set()
    start = None
    label = None

    for i, tag in enumerate(tags):
        if tag == "O":
            if label is not None:
                spans.add((label, start, i - 1))
                start, label = None, None
            continue

        prefix, role = tag.split("-", 1)

        if prefix == "B":
            if label is not None:
                spans.add((label, start, i - 1))
            start = i
            label = role

        elif prefix == "I":
            if label != role:
                # broken span, start new
                if label is not None:
                    spans.add((label, start, i - 1))
                start = i
                label = role

    if label is not None:
        spans.add((label, start, len(tags) - 1))

    return spans

def preprocess_srl_gpt2(example, tokenizer, max_length=256):
    prompt = build_srl_prompt(example["tokens"], example["predicate_index"])

    gold_tags = example["labels"]
    gold = " " + " ".join(gold_tags)

    full_text = prompt + gold

    tokenized = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    # mask prompt
    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        padding=False,
        max_length=max_length,
    )["input_ids"]

    labels = tokenized["input_ids"].copy()
    prompt_len = min(len(prompt_ids), max_length)
    labels[:prompt_len] = [-100] * prompt_len
    
    # mask padding tokens
    attn = tokenized["attention_mask"]
    labels = [lab if m == 1 else -100 for lab, m in zip(labels, attn)]

    tokenized["labels"] = labels
    return tokenized


def flatten_conll_srl(split):
    flat = []

    for doc in split:
        for sent in doc["sentences"]:
            tokens = sent["words"]

            for frame in sent["srl_frames"]:
                labels = frame["frames"]
                # Find predicate index (B-V)
                try:
                    predicate_index = labels.index("B-V")
                except ValueError:
                    continue

                flat.append({
                    "tokens": tokens,
                    "predicate_index": predicate_index,
                    "labels": labels,
                })

    return flat

def evaluate_srl(model, dataset, tokenizer, max_length=256, gen_max_new_tokens=256):
    model.eval()
    device = next(model.parameters()).device

    total_correct = 0
    total_pred = 0
    total_gold = 0

    for example in dataset:
        tokens = example["tokens"]
        gold_tags = example["labels"]
        pred_idx = example["predicate_index"]

        prompt = build_srl_prompt(tokens, pred_idx)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=min(gen_max_new_tokens, len(gold_tags) + 10),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        # parse predicted tag string
        if "SRL tags:" in decoded:
            pred_part = decoded.split("SRL tags:", 1)[1].strip()
        else:
            pred_part = decoded.strip()

        pred_tags = pred_part.split()

        # make lengths match gold for span scoring
        if len(pred_tags) < len(gold_tags):
            pred_tags = pred_tags + ["O"] * (len(gold_tags) - len(pred_tags))
        pred_tags = pred_tags[:len(gold_tags)]

        pred_spans = bio_to_spans(pred_tags)
        gold_spans = bio_to_spans(gold_tags)

        total_correct += len(pred_spans & gold_spans)
        total_pred += len(pred_spans)
        total_gold += len(gold_spans)

    precision = total_correct / total_pred if total_pred > 0 else 0.0
    recall = total_correct / total_gold if total_gold > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


        

# preprocess for POS
def preprocess_conll_pos(example, tokenizer, label_names, max_length=256):
    prompt = build_pos_prompt(example["tokens"])

    gold_tags = [label_names[i] for i in example["pos_tags"]]
    gold = " " + " ".join(gold_tags)

    full_text = prompt + gold

    tokenized = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    # mask prompt
    prompt_ids = tokenizer(prompt, 
                            truncation=True, 
                            padding=False, 
                            max_length=max_length)["input_ids"]
    labels = tokenized["input_ids"].copy()
    prompt_len = min(len(prompt_ids), max_length)
    labels[:prompt_len] = [-100] * prompt_len

    # mask padding tokens
    attn = tokenized["attention_mask"]
    labels = [lab if m == 1 else -100 for lab, m in zip(labels, attn)]

    tokenized["labels"] = labels
    return tokenized

def evaluate_conll_pos(model, dataset, tokenizer, label_names, max_length=256, gen_max_new_tokens=256):
    model.eval()
    device = next(model.parameters()).device

    outputs, targets, texts = [], [], []

    for example in dataset:
        tokens = example["tokens"]
        gold_tags = [label_names[i] for i in example["pos_tags"]]

        prompt = build_pos_prompt(tokens)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=min(gen_max_new_tokens, max_length),
                do_sample=False,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        # Take only what comes after "POS tags:"
        if "POS tags:" in decoded:
            pred_part = decoded.split("POS tags:", 1)[1].strip()
        else:
            pred_part = decoded.strip()

        pred_tags = pred_part.split()
        pred_tags = pred_tags[:len(gold_tags)]
        
        if len(pred_tags) < len(gold_tags):
            pred_tags += [""] * (len(gold_tags) - len(pred_tags))

        outputs.append(" ".join(pred_tags))
        targets.append(" ".join(gold_tags))
        texts.append(" ".join(tokens))

    precision, recall, f1 = score(texts, outputs, targets)
    return precision, recall, f1

def preprocess_conll_ner(example, tokenizer, label_names, max_length=256):
    prompt = build_ner_prompt(example["tokens"])

    gold_tags = [label_names[i] for i in example["ner_tags"]]
    gold = " " + " ".join(gold_tags)

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
    
    # mask padding tokens
    attn = tokenized["attention_mask"]
    labels = [lab if m == 1 else -100 for lab, m in zip(labels, attn)]

    tokenized["labels"] = labels
    return tokenized


def preprocess_ontonotes(example, tokenizer, label_names, id2label, max_length=256):
    prompt = build_ner_prompt(example["tokens"])

    gold_tags = [id2label[i] for i in example["tags"]]
    gold = " " + " ".join(gold_tags)

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

    # mask padding tokens
    attn = tokenized["attention_mask"]
    labels = [lab if m == 1 else -100 for lab, m in zip(labels, attn)]

    tokenized["labels"] = labels
    return tokenized


def run_conll_pos(args, seed):
    seed_everything(seed)
    print(f"\n===== Training {args.model} on CoNLL-2000 POS =====")

    raw_dataset = load_dataset("conll2000", trust_remote_code=True)

    dataset = raw_dataset["train"].train_test_split(test_size=0.1, seed=seed)
    label_names = dataset["train"].features["pos_tags"].feature.names
    num_labels = len(label_names)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # print("Preprocessing CoNLL-2000...")
    dataset = dataset.map(
        lambda x: preprocess_conll_pos(x, tokenizer, label_names, max_length=256),
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.config.pad_token_id = tokenizer.pad_token_id

    save_dir = os.path.join("checkpoints", f"{args.model.replace('/', '_')}_conll00_e{args.epochs}")
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    # print("\nStarting training...\n")
    trainer.train()

    # print("\nsaving model to", save_dir)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print("\n===== CoNLL-2000 POS Evaluation =====")
    p, r, f1 = evaluate_conll_pos(model, raw_dataset["test"], tokenizer, label_names)
    # print("CoNLL-2000 POS Results:")
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

    print("\ncompleted conll00 \n")

    return p,r,f1


def run_conll_ner(args, seed):
    seed_everything(seed)
    print(f"\n===== Training {args.model} on CoNLL-2003 NER =====")

    raw_dataset = load_dataset("conll2003", trust_remote_code=True)
    # small_train = raw_dataset["train"].select(range(100))
    # dataset = small_train.train_test_split(test_size=0.1, seed=42)

    dataset = raw_dataset["train"].train_test_split(test_size=0.1, seed=seed)
    label_names = raw_dataset["train"].features["ner_tags"].feature.names

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset.map(
        lambda x: preprocess_conll_ner(x, tokenizer, label_names, max_length=256),
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.config.pad_token_id = tokenizer.pad_token_id

    save_dir = os.path.join(
        "checkpoints", f"{args.model.replace('/', '_')}_conll03_ner_e{args.epochs}"
    )
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    trainer.train()

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print("\n===== CoNLL-2003 NER Evaluation =====")
    # p, r, f1 = evaluate_conll_ner(model, raw_dataset["test"].select(range(100)), tokenizer, label_names)
    p, r, f1 = evaluate_conll_ner(model, raw_dataset["test"], tokenizer, label_names)
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

    print("\ncompleted conll03 \n")

    return p,r,f1


def run_genia_ner(args, seed):
    seed_everything(seed)
    print(f"\n===== Training {args.model} on GENIA NER =====")

    raw_dataset = load_dataset("chufangao/GENIA-NER", trust_remote_code=True)
    # small_train = raw_dataset["train"].select(range(100))
    # dataset = small_train.train_test_split(test_size=0.1, seed=42)

    dataset = raw_dataset["train"].train_test_split(test_size=0.1, seed=seed)
    label_names = raw_dataset["train"].features["ner_tags"].feature.names

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset.map(
        lambda x: preprocess_conll_ner(x, tokenizer, label_names, max_length=256),
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.config.pad_token_id = tokenizer.pad_token_id

    save_dir = os.path.join(
        "checkpoints", f"{args.model.replace('/', '_')}_genia_ner_e{args.epochs}"
    )
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    trainer.train()

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print("\n===== GENIA NER Evaluation =====")
    p, r, f1 = evaluate_conll_ner(model, raw_dataset["test"], tokenizer, label_names)
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

    print("\ncompleted genia \n")

    return p,r,f1


def run_ontonotes_ner(args, seed):
    seed_everything(seed)
    print(f"\n===== Training {args.model} on OntoNotes5 NER =====")

    raw_dataset = load_dataset("tner/ontonotes5", trust_remote_code=True)
    # small_train = raw_dataset["train"].select(range(100))
    # dataset = small_train.train_test_split(test_size=0.1, seed=42)

    dataset = raw_dataset["train"].train_test_split(test_size=0.1, seed=seed)
    label_names = sorted({tag for seq in dataset["train"]["tags"] for tag in seq})
    num_labels = len(label_names)

    label2id = {
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
        "I-LANGUAGE": 36
        }
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset.map(
        lambda x: preprocess_ontonotes(x, tokenizer, label_names, id2label, max_length=256),
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.config.pad_token_id = tokenizer.pad_token_id

    save_dir = os.path.join(
        "checkpoints", f"{args.model.replace('/', '_')}_ontonotes5_ner_e{args.epochs}"
    )
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    trainer.train()

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print("\n===== OntoNotes5 NER Evaluation =====")
    p, r, f1 = evaluate_ontonotes_ner(model, raw_dataset["test"], tokenizer, id2label)
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

    print("\ncompleted ontonotes5 \n")

    return p,r,f1


def run_conll_srl(args, seed):
    seed_everything(seed)
    print(f"\n===== Training {args.model} on CoNLL-2012 SRL (GPT2 causal LM prompting) =====")

    # use force download if dataset error in logs
    # raw = load_dataset("ontonotes/conll2012_ontonotesv5", "english_v4", trust_remote_code=True,download_mode="force_redownload")
    raw = load_dataset("ontonotes/conll2012_ontonotesv5", "english_v4", trust_remote_code=True)
    print("loaded dataset \n\n")
    train_data = flatten_conll_srl(raw["train"])
    dev_data   = flatten_conll_srl(raw["validation"])
    test_data  = flatten_conll_srl(raw["test"])

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # add predicate markers
    tokenizer.add_special_tokens({"additional_special_tokens": ["[PRED]", "[/PRED]"]})

    train_raw = Dataset.from_list(train_data)
    dev_raw   = Dataset.from_list(dev_data)

    train_ds = train_raw.map(
        lambda x: preprocess_srl_gpt2(x, tokenizer, max_length=256),
        remove_columns=train_raw.column_names,
    )
    dev_ds = dev_raw.map(
        lambda x: preprocess_srl_gpt2(x, tokenizer, max_length=256),
        remove_columns=dev_raw.column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    save_dir = os.path.join("checkpoints", f"{args.model.replace('/', '_')}_conll12_srl_e{args.epochs}")
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

    print("\n===== CoNLL-2012 SRL Evaluation (span F1) =====")
    p, r, f1 = evaluate_srl(model, test_data, tokenizer)
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

    print("\ncompleted conll12 (srl)\n")

    return p,r,f1


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
        default="gpt2",
        help="what model to fine-tune (gpt2, checkpoints/gpt2_yelp_pr)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="conll00",
        help="what task to fine-tune ('tacred' for RE or 'conll00' for POS, 'conll03' for NER)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)

    args = parser.parse_args()

    results = []
    for i in range(10):
        if args.task == "conll00": # pos
            p,r,f1 = run_conll_pos(args,i)
        elif args.task == "conll03": # ner
            p,r,f1 = run_conll_ner(args,i)
        elif args.task == "conll12": # SRL
            p,r,f1 = run_conll_srl(args,i)
        elif args.task == "genia": # ner
            p,r,f1 = run_genia_ner(args,i)
        elif args.task == "ontonotes5": # ner
            p,r,f1 = run_ontonotes_ner(args,i)
        # elif args.task == "conll16": # OIE
        #     p,r,f1 = run_conll2016_oie(args,i)
        else:
            p,r,f1 = 0.0, 0.0, 0.0
            print("task not available")
        results.append({
            "model": args.model,
            "task": args.task,
            "run": i,
            "seed": i,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
        })
    save_csv = os.path.join(
        "outputs", "generated", f"{args.model.replace('/', '_')}_{args.task}.csv"
    )
    os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)

    write_results_csv(results, save_csv)
    print(f"\nSaved results to {save_csv}\n")