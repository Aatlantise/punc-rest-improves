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
    AutoModelForTokenClassification,
    AutoModelForSeq2SeqLM,
    EncoderDecoderModel,
)

# CaRB imports (ensure CaRB repo is in PYTHONPATH)
# run following before the py file
# export PYTHONPATH=$PYTHONPATH:./CaRB
from carb import Benchmark
from matcher import Matcher
from oie_readers.tabReader import TabReader

def seed_everything(seed: int):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def gold_to_paren_target(triples):
    # triples: [(rel, [arg1,arg2,...]), ...] to (arg1; rel;arg2) (arg1;rel;arg2)...
    clauses = []
    for rel, args in triples:
        if len(args) < 2:
            continue
        arg1 = args[0]
        rest = args[1:]
        clauses.append("(" + "; ".join([arg1, rel, *rest]) + ")")
    return " ".join(clauses)

def build_rows(oie_map):
    rows = []
    for s, triples in oie_map.items():
        tgt = gold_to_paren_target(triples).strip()
        if tgt:
            rows.append({"sentence": s, "target": tgt})
    return rows




def load_gpt2(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        # GPT2 has no pad, use eos as pad
        tokenizer.pad_token = tokenizer.eos_token

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

    decoder_config = AutoConfig.from_pretrained(model_name)
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True

    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        model_name,
        model_name,
        decoder_config=decoder_config,
    )

    # ensure special token ids exist
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.bos_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
        model.generation_config.bos_token_id = model.config.bos_token_id
        model.generation_config.eos_token_id = model.config.eos_token_id
        model.generation_config.pad_token_id = model.config.pad_token_id

    return model, tokenizer


def load_model(model_name: str):
    name = model_name.lower()
    if "gpt2" in name:
        return load_gpt2(model_name)
    elif "t5" in name:
        return load_t5(model_name)
    elif "bert" in name:
        return load_bert(model_name)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def build_prompt(sentence: str) -> str:
    # only used for GPT2
    return f"Sentence: {sentence}\nExtractions:\n"


def preprocess_example(model_name: str, tokenizer, sentence: str, target: str, max_len: int):
    name = model_name.lower()

    if "gpt2" in name:
        prompt = build_prompt(sentence)
        full_text = prompt + target

        tok_full = tokenizer(full_text, truncation=True, padding="max_length", max_length=max_len)
        tok_prompt = tokenizer(prompt, truncation=True, padding=False, max_length=max_len)

        labels = tok_full["input_ids"].copy()
        prompt_len = min(len(tok_prompt["input_ids"]), max_len)

        # ignore prompt tokens
        labels[:prompt_len] = [-100] * prompt_len

        # ignore padding tokens
        labels = [lab if attn == 1 else -100 for lab, attn in zip(labels, tok_full["attention_mask"])]

        tok_full["labels"] = labels
        return tok_full

    # seq2seq (t5 / bert2bert)
    # enc = tokenizer(sentence, truncation=True, padding="max_length", max_length=max_len)
    enc = tokenizer("extract: " + sentence, truncation=True, padding="max_length", max_length=max_len)
    dec = tokenizer(target, truncation=True, padding="max_length", max_length=max_len)
    labels = dec["input_ids"]
    labels = [x if x != tokenizer.pad_token_id else -100 for x in labels]
    enc["labels"] = labels
    return enc

def hf_map_preprocess(model_name: str, tokenizer, max_len: int):
    def _fn(ex):
        return preprocess_example(model_name, tokenizer, ex["sentence"], ex["target"], max_len)
    return _fn


def simple_collator(features):
    batch = {}
    for k in features[0].keys():
        batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)
    return batch


def read_oie_gold_file(path: str):
    """
    OIE2016 gold format
    SENT | pred_head | pred_full | arg1 | arg2 | arg3 ...

    Returns:
        dict[str, list[tuple[str, list[str]]]]
        sentence -> [(relation, [arg1,arg2,...])]
    """
    d = {}

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")

            if len(parts) < 5: # doesn't contain all needed components of setence, relation head, relation, arg1, arg2
                continue

            sent = parts[0].strip()
            rel = parts[2].strip()
            args = [p.strip() for p in parts[3:] if p.strip()]

            d.setdefault(sent, []).append((rel, args))
            
            if i == 1:
                print("Sentence number 1:", sent)
                for t in d[sent]:
                    print("  ", t)
                print()
    return d

def load_oie2016_splits(oie_dir: str):
    train_p = os.path.join(oie_dir, "train.oie")
    dev_p   = os.path.join(oie_dir, "dev.oie")
    test_p  = os.path.join(oie_dir, "test.oie")

    for p in [train_p, dev_p, test_p]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required OIE2016 file: {p}")

    train = read_oie_gold_file(train_p)
    dev   = read_oie_gold_file(dev_p)
    test  = read_oie_gold_file(test_p)

    return train, dev, test


def norm_text(s: str):
    # lowercase + collapse whitespace
    return " ".join(s.strip().lower().split())

def tuple_key(rel: str, args: list[str]):
    return (norm_text(rel),) + tuple(norm_text(a) for a in args)

def test_prf(test_map: dict, test_sents: list[str], pred_triples_per_sentence: list[list[tuple]]):
    # Exact-match P/R/F1 on ONLY the predicted test_sents.
    true_pos = false_pos = false_neg = 0

    for sent, pred_triples in zip(test_sents, pred_triples_per_sentence):
        gold_triples = test_map.get(sent, [])

        gold_keys = [tuple_key(rel, args) for (rel, args) in gold_triples]
        pred_keys = [tuple_key(rel, args) for (rel, args) in pred_triples]

        gold_used = [False] * len(gold_keys)
        pred_used = [False] * len(pred_keys)

        # exact matching
        # p for predictions, g for gold, i index, k for key
        for pi, pk in enumerate(pred_keys):
            for gi, gk in enumerate(gold_keys):
                if gold_used[gi]:
                    continue
                if pk == gk:
                    pred_used[pi] = True
                    gold_used[gi] = True
                    break

        true_pos += sum(pred_used)
        false_pos += (len(pred_used) - sum(pred_used))
        false_neg += (len(gold_used) - sum(gold_used))

    p = true_pos / (true_pos + false_pos + 1e-12)
    r = true_pos / (true_pos + false_neg + 1e-12)
    f1 = 2 * p * r / (p + r + 1e-12)
    return p, r, f1


@torch.no_grad()
def generate_one(model_name: str, model, tokenizer, sentence: str, max_len: int, max_new_tokens: int):
    device = next(model.parameters()).device
    name = model_name.lower()
    model.eval()

    if "gpt2" in name:
        prompt = build_prompt(sentence)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        gen_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=4,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return prompt, decoded

    # t5 / bert2bert
    inputs = tokenizer("extract: " + sentence, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    # gen_ids = model.generate(**inputs, num_beams=4, max_length=max_len)
    gen_ids = model.generate(**inputs, num_beams=4, max_new_tokens=max_new_tokens)
    decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return sentence, decoded


def print_one_generation(tag: str, model_name: str, model, tokenizer, rows, max_len: int, max_new_tokens: int):
    if not rows:
        print(f"\n[{tag}] No rows to generate from.\n")
        return
    ex = rows[0]
    prompt_or_sent, decoded = generate_one(model_name, model, tokenizer, ex["sentence"], max_len, max_new_tokens)
    print(f"\n===== ONE GENERATION EXAMPLE ({tag}) =====")
    print("INPUT:\n", prompt_or_sent)
    print("GOLD:\n", ex["target"])
    print("OUTPUT:\n", decoded)
    print("=========================================\n")



def parse_paren_extractions(text: str):
    """
    Parses "(arg1; rel; arg2; ...) (arg1; rel; arg2 ...)" into [(rel,[arg1,arg2,...]), ...]
    If the model output doesn't follow parentheses, this will return [].
    """
    out = []
    CLAUSE_RE = re.compile(r"\(([^()]*)\)")
    for m in CLAUSE_RE.finditer(text):
        parts = [p.strip() for p in m.group(1).split(";") if p.strip()]
        if len(parts) < 3:
            continue
        arg1 = parts[0]
        rel = parts[1]
        args = [arg1] + parts[2:]
        out.append((rel, args))
    return out


def write_carb_tab_predictions(out_path: str, sentences, pred_triples_per_sentence, conf: str = "1.0"):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for sent, triples in zip(sentences, pred_triples_per_sentence):
            for rel, args in triples:
                if len(args) < 2:
                    continue
                row = [sent, conf, rel] + args
                f.write("\t".join(row) + "\n")


def make_gold_all_file(oie_dir: str, out_path: str):
    if os.path.exists(out_path):
        return
    paths = []
    for nm in ["train.oie", "dev.oie", "test.oie"]:
        p = os.path.join(oie_dir, nm)
        if os.path.exists(p):
            paths.append(p)
    if not paths:
        raise FileNotFoundError(f"No split files to concatenate in: {oie_dir}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for p in paths:
            with open(p, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if line:
                        out.write(line + "\n")


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
        # CaRB default (typo in repo: linient)
        mfunc = Matcher.binary_linient_tuple_match

    auc, optimal = b.compare(predicted=tr.oie, matchingFunc=mfunc, output_fn=os.devnull)
    p, r, f1 = float(optimal[0]), float(optimal[1]), float(optimal[2])
    return p, r, f1, float(auc)

# ----------------------------
# Main task
# ----------------------------
def run_conll2016_oie(args, seed: int):
    seed_everything(seed)
    print(f"\n===== Training {args.model} on CoNLL-2016 / OIE2016 (quick finetune) =====")

    oie_dir = os.path.join("OIE", "oie_corpus")
    train_map, dev_map, test_map = load_oie2016_splits(oie_dir)

    # Build HF rows
    train_rows = build_rows(train_map)
    dev_rows = build_rows(dev_map)
    test_rows = build_rows(test_map)

    # quick subset for training
    rng = random.Random(seed)
    rng.shuffle(train_rows)
    # set n to 0 for full train
    if args.train_n > 0:
        train_rows = train_rows[: min(args.train_n, len(train_rows))]

    # optionally reduce dev too
    if args.dev_n > 0:
        dev_rows = dev_rows[: min(args.dev_n, len(dev_rows))]

    print(f"Train rows: {len(train_rows)} | Dev rows: {len(dev_rows)} | Test rows: {len(test_rows)}")

    train_raw = Dataset.from_list(train_rows)
    dev_raw = Dataset.from_list(dev_rows) if dev_rows else Dataset.from_list(train_rows[: min(20, len(train_rows))])
    test_raw = Dataset.from_list(test_rows)

    model, tokenizer = load_model(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("CUDA:", torch.cuda.is_available(), "device:", next(model.parameters()).device)

    # Print one generation BEFORE training
    print_one_generation("BEFORE TRAIN", args.model, model, tokenizer, test_rows, args.max_length, args.gen_max_new_tokens)

    # Tokenize datasets
    train_ds = train_raw.map(
        hf_map_preprocess(args.model, tokenizer, args.max_length),
        remove_columns=train_raw.column_names,
    )
    dev_ds = dev_raw.map(
        hf_map_preprocess(args.model, tokenizer, args.max_length),
        remove_columns=dev_raw.column_names,
    )

    save_dir = os.path.join("checkpoints", f"{args.model.replace('/', '_')}_conll16_oie_seed{seed}")
    os.makedirs(save_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=save_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True, # load best model according to eval loss
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        seed=seed,
        data_seed=seed,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=simple_collator,
    )

    trainer.train()

    # Print one generation AFTER training
    print_one_generation("AFTER TRAIN", args.model, model, tokenizer, test_rows, args.max_length, args.gen_max_new_tokens)

    # ----------------------------
    # Generate predictions on test + write CaRB .tab
    # ----------------------------
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    test_sents = list(test_map.keys())
    if args.test_n > 0:
        test_sents = test_sents[: min(args.test_n, len(test_sents))]

    pred_triples_per_sentence = []
    for idx, sent in enumerate(test_sents):
        prompt_or_sent, decoded = generate_one(args.model, model, tokenizer, sent, args.max_length, args.gen_max_new_tokens)

        # For GPT2, decoded will include the prompt sometimes; strip it
        if "gpt2" in args.model.lower() and "Extractions:" in decoded:
            decoded_after = decoded.split("Extractions:", 1)[1].strip()
        else:
            decoded_after = decoded.strip()

        # Parse to triples for CaRB tab output
        pred_triples = parse_paren_extractions(decoded_after)
        pred_triples_per_sentence.append(pred_triples)

        # Print one sample decode (first item) so you can see what the model is generating
        if idx == 5:
            print("\n===== FIFTH TEST GENERATION (raw) =====")
            print("SENT:", sent)
            print("DECODED:", decoded_after)
            print("GOLD:", gold_to_paren_target(test_map[sent]))
            print("PARSED_TRIPLES:", pred_triples)
            print("======================================\n")

    pred_path = os.path.join("outputs", "generated", f"{args.model.replace('/', '_')}_conll16_preds.tab")
    write_carb_tab_predictions(pred_path, test_sents, pred_triples_per_sentence, conf="1.0")
    print(f"Wrote predictions to: {pred_path}")
    # print("Pred file lines:", sum(1 for _ in open(pred_path, "r", encoding="utf-8")))

    # ----------------------------
    # CaRB evaluation
    # ----------------------------
    # gold_all_path = os.path.join(oie_dir, "all.oie")
    # if not os.path.exists(gold_all_path):
    #     make_gold_all_file(oie_dir, gold_all_path)

    gold_test_path = os.path.join(oie_dir, "test.oie")

    p, r, f1, auc = evaluate_carb_prf(
        # gold_oie_path=gold_all_path,
        gold_oie_path=gold_test_path,
        pred_tab_path=pred_path,
        matching=args.oie_matching,
    )

    print("\n===== CaRB Evaluation =====")
    print(f"Matching: {args.oie_matching}")
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    print("==========================\n")

    test_p, test_r, test_f1 = test_prf(
        test_map=test_map,
        test_sents=test_sents,
        pred_triples_per_sentence=pred_triples_per_sentence,)
    print("\n===== Test Set Evaluation =====")
    print(f"Precision: {test_p:.4f}, Recall: {test_r:.4f}, F1: {test_f1:.4f}")
    print("=========================================================\n")
    
    

    return p, r, f1, test_p, test_r, test_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="google-t5/t5-base",
        help="Model name or path. Examples: gpt2, google-t5/t5-base, bert-base-uncased",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)

    parser.add_argument("--train_n", type=int, default=100, help="Train on N samples (quick finetune).")
    parser.add_argument("--dev_n", type=int, default=50, help="Optionally cap dev rows (0=no cap).")
    parser.add_argument("--test_n", type=int, default=1, help="Cap test sentences (default=1 for one printed prediction).")

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--gen_max_new_tokens", type=int, default=128)

    parser.add_argument(
        "--oie_matching",
        type=str,
        default="binary_lenient",
        help="CaRB matching: binary_lenient (default), binary_strict, simple, exact, pred, lexical, strict",
    )

    args = parser.parse_args()

    # run_conll2016_oie(args, seed=0)
    for i in range(10):
        run_conll2016_oie(args, seed=i)