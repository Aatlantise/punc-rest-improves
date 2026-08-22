import os
import argparse
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModelForTokenClassification, set_seed
import random
import numpy as np

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import csv

def seed_everything(seed: int):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



# Using library f1 calculations instead of custom calculations, to be passed into trainer
def ner_compute_metrics(label_list):
    def compute_metrics(p):
        logits, labels = p
        preds = np.argmax(logits, axis=-1)

        true_predictions = []
        true_labels = []

        for pred_seq, lab_seq in zip(preds, labels):
            sent_preds = []
            sent_labs = []
            for p_id, l_id in zip(pred_seq, lab_seq):
                if l_id == -100:
                    continue
                sent_preds.append(label_list[p_id])
                sent_labs.append(label_list[l_id])
            true_predictions.append(sent_preds)
            true_labels.append(sent_labs)

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }
    return compute_metrics



# Following is custom evaluation code from T5 that may not be suitable
################################################################
def prf1(num_correct: int, num_attempted: int, num_gold: int) -> tuple[float, float, float]:
    precision = num_correct / num_attempted
    recall = num_correct / num_gold
    
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
###########################################################


def evaluate_conll_pos(model, dataset, tokenizer, label_names):
    model.eval()
    device = next(model.parameters()).device

    outputs = []
    targets = []
    texts = []

    for example in dataset:
        tokens = example["tokens"]
        gold_tags = example["pos_tags"]

        inputs = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            preds = logits.argmax(dim=-1)[0].cpu().tolist()

        word_ids = inputs.word_ids()
        pred_tags = []
        gold_seq = []

        prev_word_id = None
        for p, w_id in zip(preds, word_ids):
            if w_id is None or w_id == prev_word_id:
                continue
            pred_tags.append(label_names[p])
            gold_seq.append(label_names[gold_tags[w_id]])
            prev_word_id = w_id

        outputs.append(" ".join(pred_tags))
        targets.append(" ".join(gold_seq))
        texts.append(" ".join(tokens))

    precision, recall, f1 = score(texts, outputs, targets)
    return precision, recall, f1


# unneeded, now using library evaluation for NER, keeping here as record
# def evaluate_conll_ner(model, dataset, tokenizer, label_names):
#     model.eval()
#     device = next(model.parameters()).device

#     outputs = []
#     targets = []
#     texts = []

#     for example in dataset:
#         tokens = example["tokens"]
#         gold_tags = example["ner_tags"]

#         inputs = tokenizer(
#             tokens,
#             is_split_into_words=True,
#             return_tensors="pt",
#             truncation=True,
#             padding=True,
#         ).to(device)

#         with torch.no_grad():
#             logits = model(**inputs).logits
#             preds = logits.argmax(dim=-1)[0].cpu().tolist()

#         word_ids = inputs.word_ids()
#         pred_tags = []
#         gold_seq = []

#         prev_word_id = None
#         for p, w_id in zip(preds, word_ids):
#             if w_id is None or w_id == prev_word_id:
#                 continue
#             pred_tags.append(label_names[p])
#             gold_seq.append(label_names[gold_tags[w_id]])
#             prev_word_id = w_id

#         outputs.append(" ".join(pred_tags))
#         targets.append(" ".join(gold_seq))
#         texts.append(" ".join(tokens))

#     precision, recall, f1 = score(texts, outputs, targets)
#     return precision, recall, f1



def evaluate_ontonotes_ner(model, dataset, tokenizer, id2label):
    model.eval()
    device = next(model.parameters()).device

    outputs = []
    targets = []
    texts = []

    for example in dataset:
        tokens = example["tokens"]
        gold_tags = example["tags"]

        inputs = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            preds = logits.argmax(dim=-1)[0].cpu().tolist()

        word_ids = inputs.word_ids()
        pred_tags = []
        gold_seq = []

        prev_word_id = None
        for p, w_id in zip(preds, word_ids):
            if w_id is None or w_id == prev_word_id:
                continue
            pred_tags.append(id2label[p])
            gold_seq.append(id2label[gold_tags[w_id]])
            prev_word_id = w_id

        outputs.append(" ".join(pred_tags))
        targets.append(" ".join(gold_seq))
        texts.append(" ".join(tokens))

    precision, recall, f1 = score(texts, outputs, targets)
    return precision, recall, f1

def bio_to_spans(tags):
    """
    Convert BIO tag sequence to a set of spans.
    Each span is (label, start, end) inclusive.
    """
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

def evaluate_srl(model, dataset, tokenizer, label_names):
    model.eval()
    device = next(model.parameters()).device

    total_correct = 0
    total_pred = 0
    total_gold = 0

    for example in dataset:
        tokens = example["tokens"]
        gold_labels = example["labels"]
        pred_idx = example["predicate_index"]

        tokens = (
            tokens[:pred_idx]
            + ["[PRED]", tokens[pred_idx], "[/PRED]"]
            + tokens[pred_idx + 1 :]
        )

        inputs = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            preds = logits.argmax(dim=-1)[0].cpu().tolist()

        word_ids = inputs.word_ids()

        pred_tags = []
        gold_tags = []

        prev_word_id = None
        for p, w_id in zip(preds, word_ids):
            if w_id is None or w_id == prev_word_id:
                continue
            
            # skip predicate markers
            if tokens[w_id] in ("[PRED]", "[/PRED]"):
                prev_word_id = w_id
                continue

            if w_id < pred_idx:
                gold_id = gold_labels[w_id]
            elif w_id == pred_idx + 1:
                gold_id = gold_labels[pred_idx]
            else:
                gold_id = gold_labels[w_id - 2]

            pred_tags.append(label_names[p])  # id → string
            gold_tags.append(gold_id)         # already string

            prev_word_id = w_id
            
        pred_spans = bio_to_spans(pred_tags)
        gold_spans = bio_to_spans(gold_tags)

        total_correct += len(pred_spans & gold_spans)
        total_pred += len(pred_spans)
        total_gold += len(gold_spans)

    precision = total_correct / total_pred if total_pred > 0 else 0.0
    recall = total_correct / total_gold if total_gold > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return precision, recall, f1


def preprocess_tacred(example, tokenizer):
    tokens = example["tokens"]
    h_start, h_end = example["subj_start"], example["subj_end"]
    t_start, t_end = example["obj_start"], example["obj_end"]

    tokens = tokens.copy()

    if h_start < t_start:
        tokens.insert(h_start, "[E1]")
        tokens.insert(h_end + 2, "[/E1]")
        tokens.insert(t_start + 2, "[E2]")
        tokens.insert(t_end + 4, "[/E2]")
    else:
        tokens.insert(t_start, "[E2]")
        tokens.insert(t_end + 2, "[/E2]")
        tokens.insert(h_start + 2, "[E1]")
        tokens.insert(h_end + 4, "[/E1]")

    marked_sentence = " ".join(tokens)

    encoded = tokenizer(
        marked_sentence,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    encoded["labels"] = example["relation"]
    return encoded

#conll2000
def preprocess_conll_pos(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=256,
    )

    labels = []
    for i, word_labels in enumerate(examples["pos_tags"]):
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

        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized

#conll2003
def preprocess_conll_ner(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=256,
    )

    labels = []
    for i, word_labels in enumerate(examples["ner_tags"]):
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

        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized


def preprocess_ontonotes(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=256,
    )

    labels = []
    for i, word_labels in enumerate(examples["tags"]):
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

        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized


#conll2012 SRL
def preprocess_srl(example, tokenizer, label2id):
    tokens = example["tokens"].copy()
    labels = example["labels"].copy()
    pred = example["predicate_index"]

    tokens = (
        tokens[:pred]
        + ["[PRED]", tokens[pred], "[/PRED]"]
        + tokens[pred + 1 :]
    )

    labels = (
        labels[:pred]
        + ["O", labels[pred], "O"]
        + labels[pred + 1 :]
    )

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=256,
    )

    word_ids = encoding.word_ids()
    label_ids = []

    prev_word_id = None
    for word_id in word_ids:
        if word_id is None:
            label_ids.append(-100)
        elif word_id != prev_word_id:
            label_ids.append(label2id[labels[word_id]])
        else:
            label_ids.append(-100)
        prev_word_id = word_id

    encoding["labels"] = label_ids
    return encoding

# Also for conll2012 SRL
def flatten_conll_srl(split):
    flat = []

    for doc in split:
        for sent in doc["sentences"]:
            tokens = sent["words"]

            for frame in sent["srl_frames"]:
                labels = frame["frames"]

                # Sanity check
                assert len(tokens) == len(labels)

                # Find predicate index (B-V)
                try:
                    predicate_index = labels.index("B-V")
                except ValueError:
                    continue  # very rare, but safe

                flat.append({
                    "tokens": tokens,
                    "predicate_index": predicate_index,
                    "labels": labels,
                })

    return flat

# OIE for conll 2016
def flatten_conll2016_oie(split):
    flat = []

    for ex in split:
        tokens = ex["tokens"]

        for frame in ex["frames"]:
            labels = frame["tags"]

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

# preprocessing for OIE conll 2016
def preprocess_oie(example, tokenizer, label2id):
    tokens = example["tokens"].copy()
    labels = example["labels"].copy()
    pred = example["predicate_index"]

    tokens = (
        tokens[:pred]
        + ["[PRED]", tokens[pred], "[/PRED]"]
        + tokens[pred + 1 :]
    )

    labels = (
        labels[:pred]
        + ["O", labels[pred], "O"]
        + labels[pred + 1 :]
    )

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=256,
    )

    word_ids = encoding.word_ids()
    label_ids = []

    prev_word_id = None
    for word_id in word_ids:
        if word_id is None:
            label_ids.append(-100)
        elif word_id != prev_word_id:
            label_ids.append(label2id[labels[word_id]])
        else:
            label_ids.append(-100)
        prev_word_id = word_id

    encoding["labels"] = label_ids
    return encoding

def evaluate_oie(model, dataset, tokenizer, label_names):
    return evaluate_srl(model, dataset, tokenizer, label_names)

def run_conll2016_oie(args, seed):
    seed_everything(seed)
    print(f"\n===== Training {args.model} seed {seed} on CoNLL-2016 OIE =====")

    raw = load_dataset("conll2016", trust_remote_code=True)

    train_data = flatten_conll2016_oie(raw["train"])
    dev_data   = flatten_conll2016_oie(raw["validation"])
    test_data  = flatten_conll2016_oie(raw["test"])

    # build label space (avoid unseen in test)
    all_labels = set()
    for split in [train_data, dev_data]:
        for ex in split:
            all_labels.update(ex["labels"])

    label_list = sorted(all_labels)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[PRED]", "[/PRED]"]}
    )

    train_raw = Dataset.from_list(train_data)
    dev_raw   = Dataset.from_list(dev_data)

    train_ds = train_raw.map(
        lambda x: preprocess_oie(x, tokenizer, label2id),
        remove_columns=train_raw.column_names,
    )

    dev_ds = dev_raw.map(
        lambda x: preprocess_oie(x, tokenizer, label2id),
        remove_columns=dev_raw.column_names,
    )

    # short test
    train_ds = train_ds.select(range(min(10, len(train_ds))))
    dev_ds   = dev_ds.select(range(min(10, len(dev_ds))))
    test_data = test_data[:10]

    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    save_dir = os.path.join(
        "checkpoints", f"{args.model.replace('/', '_')}_conll2016_oie_e{args.epochs}"
    )
    os.makedirs(save_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=save_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        # tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    p, r, f1 = evaluate_oie(model, test_data, tokenizer, label_list)
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

    print("\nOIE finished\n")

    return p,r,f1


def run_tacred(args, seed):
    seed_everything(seed)
    print(f"\n===== Training {args.model} seed {seed} on TACRED =====")
    dataset = load_dataset("DFKI-SLT/tacred", trust_remote_code=True)

    label_names = dataset["train"].features["relation"].names
    num_labels = len(label_names)
    # print(f"Number of labels: {num_labels}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    tokenizer.add_special_tokens(special_tokens)

    # print("preprocessing dataset...")
    dataset = dataset.map(lambda x: preprocess_tacred(x, tokenizer),
                          remove_columns=dataset["train"].column_names,)

    # print(f"loading model {args.model}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    save_name = f"{args.model.replace('/', '_')}_tacred_epoch{args.epochs}"
    save_dir = os.path.join("checkpoints", save_name)
    os.makedirs(save_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=save_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        weight_decay=0.01,
        logging_steps=100,
        fp16=torch.cuda.is_available(),  # automatic mixed precision on GPU
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        # tokenizer=tokenizer,
    )

    # print("\nStarting training...\n")
    trainer.train()

    # print(f"\n===== Validation Results ({args.model} on TACRED) =====")
    results = trainer.evaluate()
    print(results)
    
    # print("\nsaving model to", save_dir)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    # print("\ncompleted tacred \n")

def run_conll_pos(args, seed):
    seed_everything(seed)
    print(f"\n===== Training {args.model} seed {seed} on CoNLL-2000 POS =====")

    raw_dataset = load_dataset("conll2000", trust_remote_code=True)

    dataset = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)
    label_names = dataset["train"].features["pos_tags"].feature.names
    num_labels = len(label_names)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # print("Preprocessing CoNLL-2000...")
    dataset = dataset.map(
        lambda x: preprocess_conll_pos(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    save_dir = os.path.join(
        "checkpoints", f"{args.model.replace('/', '_')}_conll00_e{args.epochs}"
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        # tokenizer=tokenizer,
    )
    # print("\nStarting training...\n")
    trainer.train()

    # print("\nsaving model to", save_dir)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


    # print("\n===== CoNLL-2000 POS Evaluation =====")
    p, r, f1 = evaluate_conll_pos(model, raw_dataset["test"], tokenizer, label_names)
    # print("CoNLL-2000 POS Results:")
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

    print("\ncompleted conll00 \n")

    return p,r,f1

def run_conll_ner(args, seed):
    seed_everything(seed)
    print(f"\n===== Training {args.model} seed {seed} on CoNLL-2003 NER =====")

    # raw_dataset = load_dataset("conll2003", trust_remote_code=True)
    dataset = load_dataset("conll2003", trust_remote_code=True)
    label_names = dataset["train"].features["ner_tags"].feature.names
    num_labels = len(label_names)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # print("Preprocessing CoNLL-2003...")
    dataset = dataset.map(
        lambda x: preprocess_conll_ner(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        # tokenizer=tokenizer,
        compute_metrics=ner_compute_metrics(label_names),
    )
    # print("\nStarting training...\n")
    trainer.train()
    
    # print(trainer.evaluate())
    test_results = trainer.evaluate(eval_dataset=dataset["test"])
    print(test_results)
    p = float(test_results.get("eval_precision", test_results.get("precision", 0.0)))
    r = float(test_results.get("eval_recall", test_results.get("recall", 0.0)))
    f1 = float(test_results.get("eval_f1", test_results.get("f1", 0.0)))

    print("\n above is default evaluation \n")
    # print("\nsaving model to", save_dir)
    # model.save_pretrained(save_dir)
    # tokenizer.save_pretrained(save_dir)

    # print("\n===== CoNLL-2003 NER Evaluation =====")
    # p, r, f1 = evaluate_conll_ner(model, raw_dataset["test"], tokenizer, label_names)
    # print("CoNLL-2003 NER Results:")
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

    print("\ncompleted conll03 \n")
    return p,r,f1

def run_conll_srl(args, seed):
    seed_everything(seed)
    print(f"\n===== Training {args.model} seed {seed} on CoNLL-2012 SRL =====")

    raw = load_dataset("ontonotes/conll2012_ontonotesv5", "english_v4", trust_remote_code=True)
    
    train_data = flatten_conll_srl(raw["train"])
    dev_data   = flatten_conll_srl(raw["validation"])
    test_data  = flatten_conll_srl(raw["test"])

    # label_list = sorted({l for ex in train_data for l in ex["labels"]})
    # above doesn't work since some labels only in test
    all_labels = set()

    for split in [train_data, dev_data]:
        for ex in split:
            all_labels.update(ex["labels"])

    label_list = sorted(all_labels)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[PRED]", "[/PRED]"]}
    )

    train_raw = Dataset.from_list(train_data)
    dev_raw   = Dataset.from_list(dev_data)

    train_ds = train_raw.map(
        lambda x: preprocess_srl(x, tokenizer, label2id),
        remove_columns=train_raw.column_names,
    )

    dev_ds = dev_raw.map(
        lambda x: preprocess_srl(x, tokenizer, label2id),
        remove_columns=dev_raw.column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    save_dir = os.path.join(
        "checkpoints", f"{args.model.replace('/', '_')}_conll_srl_e{args.epochs}"
    )
    os.makedirs(save_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=f"checkpoints/{args.model.replace('/', '_')}_srl",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
    )
    
    # print("\nStarting training...\n")
    trainer.train()

    # print("\nsaving model to", save_dir)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # print("\n===== SRL Test Evaluation =====")
    p, r, f1 = evaluate_srl(model, test_data, tokenizer, label_list)
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

    print("\nSRL finished\n")
    return p,r,f1

def run_genia_ner(args, seed):
    seed_everything(seed)
    print(f"\n===== Training {args.model} seed {seed} on GENIA NER =====")

    # raw_dataset = load_dataset("chufangao/GENIA-NER", trust_remote_code=True)
    dataset = load_dataset("chufangao/GENIA-NER", trust_remote_code=True)
    label_names = dataset["train"].features["ner_tags"].feature.names
    num_labels = len(label_names)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # print("Preprocessing GENIA NER...")
    dataset = dataset.map(
        lambda x: preprocess_conll_ner(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        # tokenizer=tokenizer,
        compute_metrics=ner_compute_metrics(label_names),
    )
    # print("\nStarting training...\n")
    trainer.train()

    # Jan 29 check fp16 or bf16
    print("fp16 enabled:", trainer.args.fp16)
    print("bf16 enabled:", trainer.args.bf16)

    # print("\nsaving model to", save_dir)
    # model.save_pretrained(save_dir)
    # tokenizer.save_pretrained(save_dir)

    test_results = trainer.evaluate(eval_dataset=dataset["test"])
    print(test_results)

    p = float(test_results.get("eval_precision", test_results.get("precision", 0.0)))
    r = float(test_results.get("eval_recall", test_results.get("recall", 0.0)))
    f1 = float(test_results.get("eval_f1", test_results.get("f1", 0.0)))


    # print("\n===== GENIA NER Evaluation =====")
    # p, r, f1 = evaluate_conll_ner(model, raw_dataset["test"], tokenizer, label_names)
    # print("GENIA NER Results:")
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

    print("\ncompleted genia_ner \n")

    return p,r,f1

def run_ontonotes_ner(args, seed):
    seed_everything(seed)
    print(f"\n===== Training {args.model} seed {seed} on ontonotes5 NER =====")

    raw_dataset = load_dataset("tner/ontonotes5", trust_remote_code=True)
    dataset = load_dataset("tner/ontonotes5", trust_remote_code=True)
    # print(raw_dataset)
    # label_names = sorted({tag for seq in dataset["train"]["tags"] for tag in seq}) # these are numbers, can't evaluate
    # print(label_names)
    # num_labels = len(label_names)

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

    label_names = [id2label[i] for i in range(len(id2label))]
    num_labels = len(label_names)

    # print(label_names)
    # print("HERE")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # print("Preprocessing ontonotes5 NER...")

    # def preprocess_ontonotes(examples):
    #     # only need to rename "tags" to "ner_tags"
    #     examples = {"tokens": examples["words"], "ner_tags": examples["tags"]}
    #     return preprocess_conll_ner(examples, tokenizer)
    
    dataset = dataset.map(
        lambda x: preprocess_ontonotes(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # quick test reduce dataset to size 5
    # dataset["train"] = dataset["train"].select(range(min(10, len(dataset["train"]))))
    # dataset["validation"] = dataset["validation"].select(range(min(10, len(dataset["validation"]))))
    # dataset["test"] = dataset["test"].select(range(min(10, len(dataset["test"]))))

    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        # tokenizer=tokenizer,
        compute_metrics=ner_compute_metrics(label_names),
    )
    # print("\nStarting training...\n")
    trainer.train()

    # print("\nsaving model to", save_dir)
    # model.save_pretrained(save_dir)
    # tokenizer.save_pretrained(save_dir)

    test_results = trainer.evaluate(eval_dataset=dataset["test"])
    print(test_results)

    p = float(test_results.get("eval_precision", test_results.get("precision", 0.0)))
    r = float(test_results.get("eval_recall", test_results.get("recall", 0.0)))
    f1 = float(test_results.get("eval_f1", test_results.get("f1", 0.0)))


    # print("\n===== ontonotes5 NER Evaluation =====")
    # p, r, f1 = evaluate_ontonotes_ner(model, raw_dataset["test"], tokenizer, id2label)
    # print("ontonotes5 NER Results:")
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

    print("\ncompleted ontonotes5_ner \n")

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
        default="bert-base-uncased",
        help="what model to fine-tune (bert-base-uncased, felflare/bert-restore-punctuation)",
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
        if args.task == "tacred":
            p,r,f1 = run_tacred(args,i)
        elif args.task == "conll00": # pos
            p,r,f1 = run_conll_pos(args,i)
        elif args.task == "conll03": # ner
            p,r,f1 = run_conll_ner(args,i)
        elif args.task == "conll12": # SRL
            p,r,f1 = run_conll_srl(args,i)
        elif args.task == "genia": # ner
            p,r,f1 = run_genia_ner(args,i)
        elif args.task == "ontonotes5": # ner
            p,r,f1 = run_ontonotes_ner(args,i)
        elif args.task == "conll16": # OIE
            p,r,f1 = run_conll2016_oie(args,i)
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
