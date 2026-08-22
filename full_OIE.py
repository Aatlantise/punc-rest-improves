import os
import glob
import argparse
import random
import numpy as np
import re
import csv

from collections import defaultdict
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    Trainer,
    set_seed,
    AutoModelForTokenClassification,
    AutoModelForSeq2SeqLM,
    EncoderDecoderModel,
    DataCollatorForSeq2Seq,
    DataCollatorForTokenClassification,
    DataCollatorForLanguageModeling,
    default_data_collator
)

# CaRB imports (ensure CaRB repo is in PYTHONPATH)
# run following before the py file
# export PYTHONPATH=$PYTHONPATH:./CaRB

from carb import Benchmark
from matcher import Matcher
from oie_readers.tabReader import TabReader

DEFAULT_CARB_TEST = "./CaRB/data/gold/test.tsv"

LABEL_LIST = [
    "O",
    "B-ARG1", "I-ARG1",
    "B-REL", "I-REL",
    "B-ARG2", "I-ARG2",
]
label2id = {x: i for i, x in enumerate(LABEL_LIST)}
id2label = {i: x for x, i in label2id.items()}

def seed_everything(seed: int):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_gpt2(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        # GPT2 has no pad, use eos as pad
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

def load_t5(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model.config.pad_token_id = tokenizer.pad_token_id

    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.pad_token_id

    return model, tokenizer

def load_bert(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model, tokenizer


def load_model(model_name: str):
    name = model_name.lower()
    if "gpt" in name:
        return load_gpt2(model_name)
    elif "t5" in name:
        return load_t5(model_name)
    elif "bert" in name:
        return load_bert(model_name)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    

def read_oie_gold_file(path: str, n: int = 0):
    """
    Reads OIE-style TSV:
        sentence \t pred_head \t relation \t arg1 \t arg2 \t arg3 ...

    Returns a list of row-level extraction dicts:
        {
            "sentence": str,
            "pred_head": str,
            "relation": str,
            "arg1": str,
            "arg2": str,
            "extra_args": list[str],
        }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing OIE file: {path}")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 5:
                # skip malformed lines
                continue

            sentence = parts[0].strip()
            pred_head = parts[1].strip()
            relation = parts[2].strip()
            arg1 = parts[3].strip()
            args = [p.strip() for p in parts[4:] if p.strip()]

            arg2 = args[0] if len(args) > 0 else ""
            extra_args = args[1:] if len(args) > 1 else []

            rows.append({
                "sentence": sentence,
                "pred_head": pred_head,
                "relation": relation,
                "arg1": arg1,
                "arg2": arg2,
                "extra_args": extra_args,
            })

            if n > 0 and len(rows) >= n:
                break

    return rows


def load_oie2016_splits(oie_dir: str, n_train: int = 0, n_dev: int = 0, n_test: int = 0):
    train_p = os.path.join(oie_dir, "train.oie")
    dev_p   = os.path.join(oie_dir, "dev.oie")
    test_p  = os.path.join(oie_dir, "test.oie")

    for p in [train_p, dev_p, test_p]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required OIE2016 file: {p}")

    train = read_oie_gold_file(train_p, n=n_train)
    dev   = read_oie_gold_file(dev_p, n=n_dev)
    test  = read_oie_gold_file(test_p, n=n_test)

    return train, dev, test


def group_oie_by_sentence(rows):
    """
    Groups row-level extractions into sentence-level examples.

    Returns:
        [{
            "sentence": str,
            "extractions": [{
                    "pred_head": ...,
                    "relation": ...,
                    "arg1": ...,
                    "arg2": ...,
                    "extra_args": [...]
                }]
            }]
    """
    grouped = defaultdict(list)

    for ex in rows:
        grouped[ex["sentence"]].append({
            "pred_head": ex["pred_head"],
            "relation": ex["relation"],
            "arg1": ex["arg1"],
            "arg2": ex["arg2"],
            "extra_args": ex["extra_args"],
        })

    sentence_examples = []
    for sentence, extractions in grouped.items():
        sentence_examples.append({
            "sentence": sentence,
            "extractions": extractions,
        })

    return sentence_examples


def make_target_text_from_extractions(extractions):
    """
    For T5 / GPT2:
    convert all triples for one sentence into a single target string.
    """
    triple_strs = []

    for ex in extractions:
        parts = [
            f"<arg1> {ex['arg1']} </arg1>",
            f"<rel> {ex['relation']} </rel>",
        ]

        if ex["arg2"]:
            parts.append(f"<arg2> {ex['arg2']} </arg2>")

        for i, extra_arg in enumerate(ex.get("extra_args", []), start=3):
            parts.append(f"<arg{i}> {extra_arg} </arg{i}>")

        triple_strs.append("<triple> " + " ".join(parts) + " </triple>")

    return " ".join(triple_strs)


def build_generation_examples(rows):
    """
    Sentence-level examples for T5 / GPT2.
        [{"sentence": str,
        "target_text": str,
        "num_extractions": int,}]
    """
    grouped = group_oie_by_sentence(rows)
    out = []

    for item in grouped:
        out.append({
            "sentence": item["sentence"],
            "target_text": make_target_text_from_extractions(item["extractions"]),
            "num_extractions": len(item["extractions"]),
        })

    return out


def build_bert_examples(rows):
    """
    Predicate-conditioned examples for BERT.

    One row = one gold extraction.
    At train time, input should be:
        sentence + [SEP] + pred_head

    Returns:
        [{
            "sentence": str,
            "predicate": str,
            "relation": str,
            "arg1": str,
            "arg2": str,
            "extra_args": list[str],
        }]
    """
    out = []
    for ex in rows:
        out.append({
            "sentence": ex["sentence"],
            "predicate": ex["pred_head"],
            "relation": ex["relation"],
            "arg1": ex["arg1"],
            "arg2": ex["arg2"],
            "extra_args": ex["extra_args"],
        })
    return out


def load_carb_test_rows(carb_test_path: str, n: int = 0):
    """
    CaRB test is used ONLY for evaluation.
    We read it with the same row parser if it follows the same TSV extraction format.
    """
    return read_oie_gold_file(carb_test_path, n=n)


def build_eval_sentence_dataset(rows):
    """
    For inference/evaluation:
    deduplicate to one item per sentence.
    """
    seen = set()
    examples = []

    for ex in rows:
        sent = ex["sentence"]
        if sent in seen:
            continue
        seen.add(sent)
        examples.append({"sentence": sent})

    return examples


def print_data_preview(train_rows, dev_rows, test_rows, name="OIE2016"):
    print(f"\n{name} row counts")
    print(f"  train rows: {len(train_rows)}")
    print(f"  dev rows:   {len(dev_rows)}")
    print(f"  test rows:  {len(test_rows)}")

    grouped_train = group_oie_by_sentence(train_rows)
    grouped_dev = group_oie_by_sentence(dev_rows)
    grouped_test = group_oie_by_sentence(test_rows)

    print(f"\n{name} sentence counts")
    print(f"  train sents: {len(grouped_train)}")
    print(f"  dev sents:   {len(grouped_dev)}")
    print(f"  test sents:  {len(grouped_test)}")

    # if len(grouped_test) > 0:
    #     sample = grouped_test[0]
    #     print("\nSample sentence:")
    #     print(sample["sentence"])
    #     print("\nSample gold extractions:")
    #     for ex in sample["extractions"][:5]:
    #         print(ex)
    if len(grouped_train) > 0:
        sample = grouped_train[0]
        print("\nSample train sentence:")
        print(sample["sentence"])
        print("\nSample train gold extractions:")
        for ex in sample["extractions"][:5]:
            print(ex)



# tokenize
def tokenize_t5_examples(raw_examples, tokenizer, max_length=256, target_max_length=256):
    dataset = Dataset.from_list(raw_examples)
    def preprocess(batch):
        inputs = [f"extract openie: {s}" for s in batch["sentence"]]
        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=target_max_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

def tokenize_gpt2_examples(raw_examples, tokenizer, max_length=256):
    dataset = Dataset.from_list(raw_examples)

    def preprocess(example):
        prompt = f"Extract OIE triples from the sentence.\nSentence: {example['sentence']}\nOutput:"
        target = f" {example['target_text']}"
        full_text = prompt + target

        tokenized_full = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        tokenized_prompt = tokenizer(
            prompt,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]
        labels = input_ids.copy()

        prompt_len = sum(1 for tok in tokenized_prompt["input_ids"] if tok != tokenizer.pad_token_id)

        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        labels = [
            -100 if tok == tokenizer.pad_token_id else lab
            for tok, lab in zip(input_ids, labels)
        ]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return dataset.map(preprocess, remove_columns=dataset.column_names)


# need for BERT
##############
def find_sublist_span(tokens, span_tokens):
    n = len(tokens)
    m = len(span_tokens)
    if m == 0:
        return None

    for i in range(n - m + 1):
        if tokens[i:i + m] == span_tokens:
            return i, i + m
    return None

def make_word_labels(sentence, arg1, relation, arg2):
    words = sentence.split()
    labels = ["O"] * len(words)

    def label_span(span_text, b_label, i_label):
        if not span_text:
            return
        span_tokens = span_text.split()
        span = find_sublist_span(words, span_tokens)
        if span is None:
            return
        start, end = span
        labels[start] = b_label
        for i in range(start + 1, end):
            labels[i] = i_label

    label_span(arg1, "B-ARG1", "I-ARG1")
    label_span(relation, "B-REL", "I-REL")
    label_span(arg2, "B-ARG2", "I-ARG2")

    return words, labels

def tokenize_bert_examples(raw_examples, tokenizer, max_length=256):
    dataset = Dataset.from_list(raw_examples)

    def preprocess(example):
        words, word_labels = make_word_labels(
            example["sentence"],
            example["arg1"],
            example["relation"],
            example["arg2"],
        )

        predicate_words = example["predicate"].split()

        tokenized = tokenizer(
            words,
            predicate_words,
            is_split_into_words=True,
            max_length=max_length,
            truncation=True,
        )

        word_ids = tokenized.word_ids(batch_index=0)
        labels = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != prev_word_id:
                labels.append(label2id[word_labels[word_id]])
            else:
                current = word_labels[word_id]
                if current.startswith("B-"):
                    current = "I-" + current[2:]
                labels.append(label2id[current])
            prev_word_id = word_id

        tokenized["labels"] = labels
        return tokenized

    return dataset.map(preprocess, remove_columns=dataset.column_names)


def print_model_example_preview(model_name, train_rows):
    lower_name = model_name.lower()

    print("\nFormatted example preview")

    if "t5" in lower_name:
        examples = build_generation_examples(train_rows)
        if not examples:
            print("No T5 examples found.")
            return

        ex = examples[0]
        print("\n[T5]")
        print("Input:")
        print(f"extract openie: {ex['sentence']}")
        print("\nTarget:")
        print(ex["target_text"])
        print(f"\nNum extractions: {ex['num_extractions']}")

    elif "gpt" in lower_name:
        examples = build_generation_examples(train_rows)
        if not examples:
            print("No GPT2 examples found.")
            return

        ex = examples[0]
        prompt = f"Extract OIE triples from the sentence.\nSentence: {ex['sentence']}\nOutput:"
        print("\n[GPT2]")
        print("Prompt:")
        print(prompt)
        print("\nTarget continuation:")
        print(ex["target_text"])
        print("\nFull training text:")
        print(prompt + " " + ex["target_text"])
        print(f"\nNum extractions: {ex['num_extractions']}")

    elif "bert" in lower_name:
        examples = build_bert_examples(train_rows)
        if not examples:
            print("No BERT examples found.")
            return

        ex = examples[0]
        words, labels = make_word_labels(
            ex["sentence"],
            ex["arg1"],
            ex["relation"],
            ex["arg2"],
        )

        print("\n[BERT]")
        print("Sentence:")
        print(ex["sentence"])
        print("\nPredicate:")
        print(ex["predicate"])
        print("\nGold fields:")
        print({
            "arg1": ex["arg1"],
            "relation": ex["relation"],
            "arg2": ex["arg2"],
            "extra_args": ex["extra_args"],
        })
        print("\nWord-level labels:")
        for w, lab in zip(words, labels):
            print(f"{w}\t{lab}")

    else:
        print(f"No preview implemented for model: {model_name}")

def build_train_and_dev_datasets(model_name, tokenizer, train_rows, dev_rows, max_length=256, target_max_length=256):
    lower_name = model_name.lower()

    if "t5" in lower_name:
        train_examples = build_generation_examples(train_rows)
        dev_examples = build_generation_examples(dev_rows)

        train_dataset = tokenize_t5_examples(
            train_examples,
            tokenizer,
            max_length=max_length,
            target_max_length=target_max_length,
        )
        dev_dataset = tokenize_t5_examples(
            dev_examples,
            tokenizer,
            max_length=max_length,
            target_max_length=target_max_length,
        )

    elif "gpt" in lower_name:
        train_examples = build_generation_examples(train_rows)
        dev_examples = build_generation_examples(dev_rows)

        train_dataset = tokenize_gpt2_examples(
            train_examples,
            tokenizer,
            max_length=max_length,
        )
        dev_dataset = tokenize_gpt2_examples(
            dev_examples,
            tokenizer,
            max_length=max_length,
        )

    elif "bert" in lower_name:
        train_examples = build_bert_examples(train_rows)
        dev_examples = build_bert_examples(dev_rows)

        train_dataset = tokenize_bert_examples(
            train_examples,
            tokenizer,
            max_length=max_length,
        )
        dev_dataset = tokenize_bert_examples(
            dev_examples,
            tokenizer,
            max_length=max_length,
        )

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return train_dataset, dev_dataset

def train_model(
    model,
    tokenizer,
    train_dataset,
    dev_dataset,
    output_dir,
    model_name,
    epochs=3,
    batch_size=8,
    lr=5e-5,
    seed=42,
):
    lower_name = model_name.lower()

    if "gpt" in lower_name:
        data_collator = default_data_collator
    elif "t5" in lower_name:
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    elif "bert" in lower_name and "roberta" not in lower_name:
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch" if dev_dataset is not None else "no",
        save_strategy="epoch" if dev_dataset is not None else "no",
        logging_strategy="epoch",
        load_best_model_at_end=True if dev_dataset is not None else False,
        metric_for_best_model="eval_loss" if dev_dataset is not None else None,
        greater_is_better=False,
        seed=seed,
        report_to="none",
        save_total_limit=1,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer



TRIPLE_RE = re.compile(
    r"(?:<\s*)?triple\s*>(.*?)(?:<\s*)?/\s*triple\s*>",
    re.DOTALL | re.IGNORECASE,
)

FIELD_RE = re.compile(
    r"(?:<\s*)?(arg\s*\d+|rel)\s*>\s*(.*?)\s*(?:<\s*)?/\s*\1\s*>",
    re.DOTALL | re.IGNORECASE,
)

def parse_generated_oie_text(text: str):
    triples = []

    for triple_match in TRIPLE_RE.finditer(text):
        triple_text = triple_match.group(1)
        fields = {}

        for tag, value in FIELD_RE.findall(triple_text):
            norm_tag = tag.lower().replace(" ", "")
            fields[norm_tag] = " ".join(value.split())

        if "arg1" in fields and "rel" in fields:
            triples.append(fields)

    return triples


def write_carb_prediction_file(predictions_by_sentence, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for sentence, extractions in predictions_by_sentence.items():
            for ex in extractions:
                conf = ex.get("confidence", 1.0)

                row = [
                    sentence,
                    str(conf),
                    ex.get("rel", ""),
                    ex.get("arg1", ""),
                ]

                arg_idx = 2
                while True:
                    key = f"arg{arg_idx}"
                    if key not in ex:
                        break
                    row.append(ex[key])
                    arg_idx += 1

                f.write("\t".join(row) + "\n")


def generate_seq2seq_predictions(model, tokenizer, sentences, model_name, batch_size=8, max_input_length=256, max_new_tokens=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    preds = {}

    for start in range(0, len(sentences), batch_size):
        batch_sents = sentences[start:start + batch_size]

        if "t5" in model_name.lower():
            inputs = [f"extract openie: {s}" for s in batch_sents]
        else:
            inputs = [f"Extract OIE triples from the sentence.\nSentence: {s}\nOutput:" for s in batch_sents]

        enc = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(device)

        gen_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=4 if "t5" in model_name.lower() else 1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        if "t5" in model_name.lower() and start == 0:
            print("\nSample T5 generated output:")
            print(decoded[0])
            print("\nParsed triples from sample:")
            print(parse_generated_oie_text(decoded[0]))

        for sent, text in zip(batch_sents, decoded):
            if "gpt" in model_name.lower():
                prompt = f"Extract OIE triples from the sentence.\nSentence: {sent}\nOutput:"
                if text.startswith(prompt):
                    text = text[len(prompt):].strip()

            triples = parse_generated_oie_text(text)
            preds[sent] = []
            for t in triples:
                ex = {"confidence": 1.0, "arg1": t.get("arg1", ""), "rel": t.get("rel", "")}
                for k, v in t.items():
                    if k.startswith("arg") and k != "arg1":
                        ex[k] = v
                preds[sent].append(ex)

    return preds


def decode_bert_tags(words, tags):
    spans = {"ARG1": [], "REL": [], "ARG2": []}
    current_type = None
    current_tokens = []

    def flush():
        nonlocal current_type, current_tokens
        if current_type and current_tokens:
            spans[current_type].append(" ".join(current_tokens))
        current_type = None
        current_tokens = []

    for word, tag in zip(words, tags):
        if tag == "O":
            flush()
            continue

        prefix, label_type = tag.split("-", 1)
        if prefix == "B":
            flush()
            current_type = label_type
            current_tokens = [word]
        elif prefix == "I" and current_type == label_type:
            current_tokens.append(word)
        else:
            flush()

    flush()

    arg1 = spans["ARG1"][0] if spans["ARG1"] else ""
    rel = spans["REL"][0] if spans["REL"] else ""
    arg2 = spans["ARG2"][0] if spans["ARG2"] else ""

    if arg1 and rel:
        return {"confidence": 1.0, "arg1": arg1, "rel": rel, "arg2": arg2}
    return None


def generate_bert_predictions(model, tokenizer, rows, max_length=256):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    predictions_by_sentence = defaultdict(list)

    with torch.no_grad():
        for ex in rows:
            words = ex["sentence"].split()
            predicate_words = ex["pred_head"].split()

            tokenized = tokenizer(
                words,
                predicate_words,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)

            outputs = model(**tokenized)
            pred_ids = outputs.logits.argmax(dim=-1)[0].tolist()

            word_ids = tokenized.word_ids(batch_index=0)
            word_level_tags = []
            prev_word_id = None

            for tok_idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                if word_id != prev_word_id:
                    word_level_tags.append(id2label[pred_ids[tok_idx]])
                prev_word_id = word_id

            extraction = decode_bert_tags(words, word_level_tags)
            if extraction is not None:
                predictions_by_sentence[ex["sentence"]].append(extraction)

    return predictions_by_sentence


def extract_prf(optimal_f1_point):
    """
    Tries common CaRB return formats.
    """
    if hasattr(optimal_f1_point, "precision"):
        return optimal_f1_point.precision, optimal_f1_point.recall, optimal_f1_point.f1

    if isinstance(optimal_f1_point, dict):
        return (
            optimal_f1_point.get("precision"),
            optimal_f1_point.get("recall"),
            optimal_f1_point.get("f1"),
        )

    if isinstance(optimal_f1_point, (list, tuple)) and len(optimal_f1_point) >= 3:
        return optimal_f1_point[0], optimal_f1_point[1], optimal_f1_point[2]

    raise ValueError(f"Could not unpack precision/recall/f1 from: {optimal_f1_point}")


def evaluate_predictions_only(gold_path, pred_path, txt_out_path=None):
    benchmark = Benchmark(gold_path)

    tr = TabReader()
    tr.read(pred_path)

    predicted = tr.oie
    if predicted is None:
        raise ValueError(f"TabReader produced no predictions from: {pred_path}")

    if txt_out_path is None:
        txt_out_path = os.devnull

    os.makedirs(os.path.dirname(txt_out_path) or ".", exist_ok=True)

    auc, optimal_f1_point = benchmark.compare(
        predicted=predicted,
        matchingFunc=Matcher.binary_linient_tuple_match,
        output_fn=txt_out_path,
    )

    p = float(optimal_f1_point[0])
    r = float(optimal_f1_point[1])
    f1 = float(optimal_f1_point[2])
    return p, r, f1


def run_test(args, model, tokenizer, test_rows):
    # safe_model_name = os.path.basename(args.model.rstrip("/")).replace("/", "_")
    pred_dir = "./outputs/generated"
    os.makedirs(pred_dir, exist_ok=True)

    model_name = os.path.basename(args.model.rstrip("/")).replace("/", "_")
    lower_name = args.model.lower()

    # ---------- CoNLL2016 ----------
    if "t5" in lower_name or "gpt" in lower_name:
        conll_sentences = [x["sentence"] for x in build_eval_sentence_dataset(test_rows)]
        conll_preds = generate_seq2seq_predictions(
            model,
            tokenizer,
            conll_sentences,
            args.model,
            batch_size=args.batch,
            max_input_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
        )
    elif "bert" in lower_name:
        conll_preds = generate_bert_predictions(
            model,
            tokenizer,
            test_rows,
            max_length=args.max_length,
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    conll_pred_path = os.path.join(pred_dir, f"{model_name}_conll16.tsv")
    conll_txt_path = os.path.join(pred_dir, f"{model_name}_conll16.txt")

    write_carb_prediction_file(conll_preds, conll_pred_path)

    print(f"Wrote CoNLL predictions to: {conll_pred_path}")
    print(f"Num CoNLL sentences with predictions: {sum(1 for v in conll_preds.values() if v)} / {len(conll_preds)}")

    conll_p, conll_r, conll_f1 = evaluate_predictions_only(
        os.path.join(args.oie_dir, "test.oie"),
        conll_pred_path,
        txt_out_path=conll_txt_path,
    )

    print("\nCoNLL2016 test")
    print(f"Precision: {conll_p:.4f}")
    print(f"Recall:    {conll_r:.4f}")
    print(f"F1:        {conll_f1:.4f}")

    # ---------- CaRB ----------
    if not os.path.exists(DEFAULT_CARB_TEST):
        raise FileNotFoundError(f"Missing CaRB gold file: {DEFAULT_CARB_TEST}")
    carb_rows = load_carb_test_rows(DEFAULT_CARB_TEST)
    

    if "t5" in lower_name or "gpt" in lower_name:
        carb_sentences = [x["sentence"] for x in build_eval_sentence_dataset(carb_rows)]
        carb_preds = generate_seq2seq_predictions(
            model,
            tokenizer,
            carb_sentences,
            args.model,
            batch_size=args.batch,
            max_input_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        carb_preds = generate_bert_predictions(
            model,
            tokenizer,
            carb_rows,
            max_length=args.max_length,
        )

    carb_pred_path = os.path.join(pred_dir, f"{model_name}_carb.tsv")
    carb_txt_path = os.path.join(pred_dir, f"{model_name}_carb.txt")

    write_carb_prediction_file(carb_preds, carb_pred_path)

    print(f"Wrote CaRB predictions to: {carb_pred_path}")
    print(f"Num CaRB sentences with predictions: {sum(1 for v in carb_preds.values() if v)} / {len(carb_preds)}")

    carb_p, carb_r, carb_f1 = evaluate_predictions_only(
        DEFAULT_CARB_TEST,
        carb_pred_path,
        txt_out_path=carb_txt_path,
    )

    print("\nCaRB test")
    print(f"Precision: {carb_p:.4f}")
    print(f"Recall:    {carb_r:.4f}")
    print(f"F1:        {carb_f1:.4f}")

    return conll_p, conll_r, conll_f1, carb_p, carb_r, carb_f1

def append_results_csv(csv_path, model_name, seed, conll_p, conll_r, conll_f1, carb_p, carb_r, carb_f1):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "model",
                "seed",
                "conll_precision",
                "conll_recall",
                "conll_f1",
                "carb_precision",
                "carb_recall",
                "carb_f1",
            ])

        writer.writerow([
            model_name,
            seed,
            conll_p,
            conll_r,
            conll_f1,
            carb_p,
            carb_r,
            carb_f1,
        ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="google-t5/t5-base",
        help="Model name or path. Examples: gpt2, google-t5/t5-base, bert-base-uncased",
    )
    parser.add_argument("--oie_dir", type=str, required=True, help="Directory containing train.oie, dev.oie, test.oie")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    # parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--target_max_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=128)

    parser.add_argument("--n_train", type=int, default=0, help="Optional cap for train rows; 0 = all")
    parser.add_argument("--n_dev", type=int, default=0, help="Optional cap for dev rows; 0 = all")
    parser.add_argument("--n_test", type=int, default=0, help="Optional cap for test rows; 0 = all")

    args = parser.parse_args()

    csv_name = f"{os.path.basename(args.output_dir.rstrip('/'))}.csv"
    results_csv = os.path.join("./outputs/generated", csv_name)

    for i in range(10):
        seed_everything(i)

        train_rows, dev_rows, test_rows = load_oie2016_splits(
            args.oie_dir,
            n_train=args.n_train,
            n_dev=args.n_dev,
            n_test=args.n_test,
        )


        print_data_preview(train_rows, dev_rows, test_rows, name="OIE2016")
        print_model_example_preview(args.model, train_rows)

        model, tokenizer = load_model(args.model)

        train_dataset, dev_dataset = build_train_and_dev_datasets(
            model_name=args.model,
            tokenizer=tokenizer,
            train_rows=train_rows,
            dev_rows=dev_rows,
            max_length=args.max_length,
            target_max_length=args.target_max_length,
        )

        print("\nDataset sizes after tokenization")
        print(f"  train: {len(train_dataset)}")
        print(f"  dev:   {len(dev_dataset)}")

        trainer = train_model(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            output_dir=args.output_dir,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            seed=i,
        )

        print("\nTraining complete.")
        print(f"Saved model to: {args.output_dir}")

        conll_p, conll_r, conll_f1, carb_p, carb_r, carb_f1 = run_test(args, model, tokenizer, test_rows)
        # write to csv, seed, p,r,f1 (for conll), p,r,f1 (for carb)
        
        append_results_csv(
            results_csv,
            os.path.basename(args.output_dir.rstrip("/")),
            i,
            conll_p, conll_r, conll_f1,
            carb_p, carb_r, carb_f1,
        )
    print(f"finished {args.model}")
