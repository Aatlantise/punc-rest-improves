import os
import argparse
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModelForTokenClassification, set_seed
import random
import numpy as np

# def prf1(num_correct: int, num_attempted: int, num_gold: int) -> tuple[float, float, float]:
#     precision = num_correct / num_attempted
#     recall = num_correct / num_gold
    
#     if precision + recall == 0:
#         f1 = 0
#     else:
#         f1 = 2 * precision * recall / (precision + recall)
#     return precision, recall, f1

# def list_hamming_dist(a, b):
#     """Count number of index matches"""
#     return sum([1 if a[i] == b[i] else 0 for i in range(min(len(a), len(b)))])



# def score(texts: list[str], outputs: list[str], targets: list[str], strict = False) -> tuple[float, float, float]:
#     """Score POS by matching"""
#     num_correct, num_attempted, num_gold = 0, 0, 0
#     for text, output, target in zip(texts, outputs, targets):
#         output_tags, target_tags = output.split(), target.split()
#         num_attempted += len(output_tags)
#         num_gold += len(target_tags)
#         num_correct += list_hamming_dist(output_tags, target_tags)
#     return prf1(num_correct, num_attempted, num_gold)

# #conll2012 SRL
# def preprocess_srl(example, tokenizer, label2id):
#     tokens = example["tokens"].copy()
#     labels = example["labels"].copy()
#     pred = example["predicate_index"]

#     tokens = (
#         tokens[:pred]
#         + ["[PRED]", tokens[pred], "[/PRED]"]
#         + tokens[pred + 1 :]
#     )

#     labels = (
#         labels[:pred]
#         + ["O", labels[pred], "O"]
#         + labels[pred + 1 :]
#     )

#     encoding = tokenizer(
#         tokens,
#         is_split_into_words=True,
#         truncation=True,
#         padding="max_length",
#         max_length=256,
#     )

#     word_ids = encoding.word_ids()
#     label_ids = []

#     prev_word_id = None
#     for word_id in word_ids:
#         if word_id is None:
#             label_ids.append(-100)
#         elif word_id != prev_word_id:
#             label_ids.append(label2id[labels[word_id]])
#         else:
#             label_ids.append(-100)
#         prev_word_id = word_id

#     encoding["labels"] = label_ids
#     return encoding

# # Also for conll2012 SRL
# def flatten_conll_srl(split):
#     flat = []

#     for doc in split:
#         for sent in doc["sentences"]:
#             tokens = sent["words"]

#             for frame in sent["srl_frames"]:
#                 labels = frame["frames"]

#                 # Sanity check
#                 assert len(tokens) == len(labels)

#                 # Find predicate index (B-V)
#                 try:
#                     predicate_index = labels.index("B-V")
#                 except ValueError:
#                     continue  # very rare, but safe

#                 flat.append({
#                     "tokens": tokens,
#                     "predicate_index": predicate_index,
#                     "labels": labels,
#                 })

#     return flat

# def bio_to_spans(tags):
#     """
#     Convert BIO tag sequence to a set of spans.
#     Each span is (label, start, end) inclusive.
#     """
#     spans = set()
#     start = None
#     label = None

#     for i, tag in enumerate(tags):
#         if tag == "O":
#             if label is not None:
#                 spans.add((label, start, i - 1))
#                 start, label = None, None
#             continue

#         prefix, role = tag.split("-", 1)

#         if prefix == "B":
#             if label is not None:
#                 spans.add((label, start, i - 1))
#             start = i
#             label = role

#         elif prefix == "I":
#             if label != role:
#                 # broken span, start new
#                 if label is not None:
#                     spans.add((label, start, i - 1))
#                 start = i
#                 label = role

#     if label is not None:
#         spans.add((label, start, len(tags) - 1))

#     return spans

# def evaluate_srl(model, dataset, tokenizer, label_names):
#     model.eval()
#     device = next(model.parameters()).device

#     total_correct = 0
#     total_pred = 0
#     total_gold = 0

#     for example in dataset:
#         tokens = example["tokens"]
#         gold_labels = example["labels"]
#         pred_idx = example["predicate_index"]

#         tokens = (
#             tokens[:pred_idx]
#             + ["[PRED]", tokens[pred_idx], "[/PRED]"]
#             + tokens[pred_idx + 1 :]
#         )

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
#         gold_tags = []

#         prev_word_id = None
#         for p, w_id in zip(preds, word_ids):
#             if w_id is None or w_id == prev_word_id:
#                 continue

#             if tokens[w_id] in ("[PRED]", "[/PRED]"):
#                 prev_word_id = w_id
#                 continue

#             if w_id < pred_idx:
#                 gold_id = gold_labels[w_id]
#             elif w_id == pred_idx + 1:
#                 gold_id = gold_labels[pred_idx]
#             else:
#                 gold_id = gold_labels[w_id - 2]

#             pred_tags.append(label_names[p])  # id → string
#             gold_tags.append(gold_id)         # already string

#             prev_word_id = w_id

#         pred_spans = bio_to_spans(pred_tags)
#         gold_spans = bio_to_spans(gold_tags)

#         total_correct += len(pred_spans & gold_spans)
#         total_pred += len(pred_spans)
#         total_gold += len(gold_spans)

#     precision = total_correct / total_pred if total_pred > 0 else 0.0
#     recall = total_correct / total_gold if total_gold > 0 else 0.0
#     f1 = (
#         2 * precision * recall / (precision + recall)
#         if precision + recall > 0
#         else 0.0
#     )

#     return precision, recall, f1



# def preprocess_ontonotes(examples, tokenizer, label2id):
#     tokenized = tokenizer(
#         examples["tokens"],
#         truncation=True,
#         is_split_into_words=True,
#         padding="max_length",
#         max_length=256,
#     )

#     labels = []
#     for i, word_labels in enumerate(examples["tags"]):
#         word_ids = tokenized.word_ids(batch_index=i)
#         previous_word_id = None
#         label_ids = []

#         for word_id in word_ids:
#             if word_id is None:
#                 label_ids.append(-100)
#             elif word_id != previous_word_id:
#                 label_ids.append(word_labels[word_id])
#             else:
#                 label_ids.append(-100)

#             previous_word_id = word_id

#         labels.append(label_ids)

#     tokenized["labels"] = labels
#     return tokenized


# def evaluate_conll_ner(model, dataset, tokenizer, label_names):
#     model.eval()
#     device = next(model.parameters()).device

#     outputs = []
#     targets = []
#     texts = []

#     for example in dataset:
#         tokens = example["tokens"]
#         gold_tags = example["tags"]

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
#             # pred_tags.append(label_names[p])
#             # gold_seq.append(label_names[gold_tags[w_id]])
#             # prev_word_id = w_id
#             pred_tags.append(id2label[p])
#             gold_seq.append(id2label[gold_tags[w_id]])
#             prev_word_id = w_id

#         outputs.append(" ".join(pred_tags))
#         targets.append(" ".join(gold_seq))
#         texts.append(" ".join(tokens))

#     precision, recall, f1 = score(texts, outputs, targets)
#     return precision, recall, f1





# # checkpoints = ["checkpoints/felflare_bert-restore-punctuation_srl", "checkpoints/bert-base-uncased_srl"]
# checkpoints = ["checkpoints/felflare_bert-restore-punctuation_ontonotes5_ner_e5", "checkpoints/bert-base-uncased_ontonotes5_ner_e5"]
# # model_dir = "checkpoints/felflare_bert-restore-punctuation_srl" # bert-base-uncased_srl
# for model_dir in checkpoints:
#     print(f"Evaluating model from {model_dir}")
#     # tokenizer = AutoTokenizer.from_pretrained(model_dir)
#     # model = AutoModelForTokenClassification.from_pretrained(model_dir)

#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # model.to(device)
#     # model.eval()

#     # raw = load_dataset("ontonotes/conll2012_ontonotesv5", "english_v4", trust_remote_code=True)

#     # test_data = flatten_conll_srl(raw["test"])

#     # label_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]

#     # p, r, f1 = evaluate_srl(model, test_data, tokenizer, label_names)

#     # print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
#     # print(f"\nfinished evaluating {model_dir}\n")

#     raw_dataset = load_dataset("tner/ontonotes5", trust_remote_code=True)
#     dataset = load_dataset("tner/ontonotes5", trust_remote_code=True)
#     # print(raw_dataset)
#     label_names = sorted({tag for seq in dataset["train"]["tags"] for tag in seq})
#     num_labels = len(label_names)

#     label2id = {
#         "O": 0,
#         "B-CARDINAL": 1,
#         "B-DATE": 2,
#         "I-DATE": 3,
#         "B-PERSON": 4,
#         "I-PERSON": 5,
#         "B-NORP": 6,
#         "B-GPE": 7,
#         "I-GPE": 8,
#         "B-LAW": 9,
#         "I-LAW": 10,
#         "B-ORG": 11,
#         "I-ORG": 12, 
#         "B-PERCENT": 13,
#         "I-PERCENT": 14, 
#         "B-ORDINAL": 15, 
#         "B-MONEY": 16, 
#         "I-MONEY": 17, 
#         "B-WORK_OF_ART": 18, 
#         "I-WORK_OF_ART": 19, 
#         "B-FAC": 20, 
#         "B-TIME": 21, 
#         "I-CARDINAL": 22, 
#         "B-LOC": 23, 
#         "B-QUANTITY": 24, 
#         "I-QUANTITY": 25, 
#         "I-NORP": 26, 
#         "I-LOC": 27, 
#         "B-PRODUCT": 28, 
#         "I-TIME": 29, 
#         "B-EVENT": 30,
#         "I-EVENT": 31,
#         "I-FAC": 32,
#         "B-LANGUAGE": 33,
#         "I-PRODUCT": 34,
#         "I-ORDINAL": 35,
#         "I-LANGUAGE": 36
#         }
#     id2label = {i: label for label, i in label2id.items()}

#     print(id2label)

#     print(label_names)
#     print("HERE")

#     tokenizer = AutoTokenizer.from_pretrained(model_dir)

#     print("Preprocessing ontonotes5 NER...")

#     # def preprocess_ontonotes(examples):
#     #     # only need to rename "tags" to "ner_tags"
#     #     examples = {"tokens": examples["words"], "ner_tags": examples["tags"]}
#     #     return preprocess_conll_ner(examples, tokenizer)
    
#     dataset = dataset.map(
#         lambda x: preprocess_ontonotes(x, tokenizer, label2id),
#         batched=True,
#         remove_columns=dataset["train"].column_names,
#     )

#     model = AutoModelForTokenClassification.from_pretrained(
#         model_dir,
#         num_labels=num_labels,
#         ignore_mismatched_sizes=True,
#     )


#     print("\n===== ontonotes5 NER Evaluation =====")
#     p, r, f1 = evaluate_conll_ner(model, raw_dataset["test"], tokenizer, id2label)
#     print("ontonotes5 NER Results:")
#     print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

#     print("\ncompleted ontonotes5_ner \n")




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
        #     print(f"the value is {p}, {id2label[p]} \n")

        # print("HERE \n")
        # print(id2label)

        outputs.append(" ".join(pred_tags))
        targets.append(" ".join(gold_seq))
        texts.append(" ".join(tokens))

    precision, recall, f1 = score(texts, outputs, targets)
    return precision, recall, f1



checkpoints = ["checkpoints/felflare_bert-restore-punctuation_ontonotes5_ner_e5", "checkpoints/bert-base-uncased_ontonotes5_ner_e5"]
# model_dir = "checkpoints/felflare_bert-restore-punctuation_srl" # bert-base-uncased_srl
for model_dir in checkpoints:
    raw_dataset = load_dataset("tner/ontonotes5", trust_remote_code=True)
    dataset = load_dataset("tner/ontonotes5", trust_remote_code=True)
    # print(raw_dataset)
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
    print(id2label)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    dataset = dataset.map(
        lambda x: preprocess_ontonotes(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_dir,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    print("\n===== ontonotes5 NER Evaluation =====")
    p, r, f1 = evaluate_ontonotes_ner(model, raw_dataset["test"], tokenizer, id2label)
    print("ontonotes5 NER Results:")
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

    print("\ncompleted ontonotes5_ner \n")