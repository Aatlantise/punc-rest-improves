import argparse
import contextlib
import csv
import fcntl
import gc
import itertools
import json
import math
import random
import re
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import auc
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, BertModel
from transformers import get_linear_schedule_with_warmup
from transformers import __version__ as transformers_version


WEIGHT_FILENAMES = {
    "model.safetensors", "model.safetensors.index.json",
    "pytorch_model.bin", "pytorch_model.bin.index.json",
}
# BIO labels are fixed here, not inferred from test data.
# ARG0 is the first .oie argument, ARG1 the second, etc.
PRED_LABELS = ["O", "B-PRED", "I-PRED"]
BIO_LABELS = [
    "O",
    "B-REL", "I-REL",
    "B-ARG0", "I-ARG0",
    "B-ARG1", "I-ARG1",
    "B-ARG2", "I-ARG2",
    "B-ARG3", "I-ARG3",
    "B-ARG4", "I-ARG4",
    "B-ARG5", "I-ARG5",
    "B-ARG6", "I-ARG6",
]
# OIE2016 has up to seven arguments. If using another corpus, extend this list.

# Split words and punctuation for BIO span alignment; keep original offsets.
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(value.rstrip("/")).name or value)


def resolve_model_path(value):
    path = Path(value).expanduser()
    if not path.exists():
        if value.startswith((".", "/", "checkpoints/")):
            raise FileNotFoundError(f"Model path does not exist: {value}")
        return value
    if not path.is_dir():
        raise ValueError(f"Expected a model directory: {path}")
    def has_weights(p):
        return p.is_dir() and any((p / n).is_file() for n in WEIGHT_FILENAMES)
    if has_weights(path):
        return str(path.resolve())
    candidates = [p for p in path.rglob("*") if has_weights(p)]
    if not candidates:
        raise FileNotFoundError(f"No Hugging Face weights under {path}")
    numbered = [(int(p.name.split("-")[-1]), p) for p in candidates if re.fullmatch(r"checkpoint-\d+", p.name)]
    chosen = max(numbered, key=lambda x: x[0])[1] if numbered else max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"[checkpoint resolver] {path} -> {chosen}")
    return str(chosen.resolve())


def resolve_repo_layout(args):
    """Accept either --benchmark-repo OIE or OIE/oie-benchmark."""
    root = Path(args.benchmark_repo or "OIE").expanduser()
    candidates = [root, root / "oie-benchmark"]
    if args.benchmark_repo is None:
        candidates += [Path("oie-benchmark"), Path(".")]
    repo = next((p for p in candidates if (p / "benchmark.py").is_file()
                 and (p / "matcher.py").is_file() and (p / "oie_readers").is_dir()), None)
    if repo is None:
        raise FileNotFoundError(
            f"Cannot find oie-benchmark under {root}; pass its directory or its parent."
        )
    data_candidates = [root / "oie_corpus", repo.parent / "oie_corpus", repo / "oie_corpus"]
    data = Path(args.data_dir).expanduser() if args.data_dir else next(
        (p for p in data_candidates
         if all((p / f"{split}.oie").is_file() for split in ("train", "dev", "test"))), None
    )
    if data is None:
        raise FileNotFoundError("Cannot find train.oie/dev.oie/test.oie; pass --data-dir.")
    for split in ("train", "dev", "test"):
        if not (data / f"{split}.oie").is_file():
            raise FileNotFoundError(data / f"{split}.oie")
    args.benchmark_repo = str(repo.resolve())
    args.data_dir = str(data.resolve())


def tokenize_text(text):
    matches = list(TOKEN_PATTERN.finditer(text))
    return [m.group() for m in matches], [(m.start(), m.end()) for m in matches]


def normalize_token(token):
    # Case-insensitive alignment only. Output text keeps the source spelling.
    return token.lower()


def normalize_tokens(tokens):
    return [normalize_token(t) for t in tokens]


def read_oie_split(path):
    grouped = {}
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 4:
                raise ValueError(f"{path}:{line_number}: expected sentence, base relation, full relation, arguments")
            sentence, base, relation = [s.strip() for s in fields[:3]]
            args = [s.strip() for s in fields[3:] if s.strip()]
            if not sentence or not base or not relation or not args:
                raise ValueError(f"{path}:{line_number}: empty sentence/predicate/arguments")
            if sentence not in grouped:
                words, offsets = tokenize_text(sentence)
                grouped[sentence] = {"source": sentence, "tokens": words, "offsets": offsets, "extractions": []}
            grouped[sentence]["extractions"].append({"base_predicate": base, "predicate": relation, "args": args})
    if not grouped:
        raise ValueError(f"Empty OIE split: {path}")
    return list(grouped.values())


def find_spans(haystack, needle):
    if not needle:
        return []
    n = len(needle)
    return [(i, i + n) for i in range(len(haystack) - n + 1) if haystack[i:i + n] == needle]


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def span_gap(a, b):
    return max(0, a[0] - b[1], b[0] - a[1])


def align_relation(source, relation, anchor, argument_spans):
    # Longest contiguous gold-relation subphrase containing the source anchor.
    best = anchor
    for size in range(len(relation), 0, -1):
        candidates = []
        for begin in range(len(relation) - size + 1):
            for span in find_spans(source, relation[begin:begin + size]):
                if span[0] <= anchor[0] and span[1] >= anchor[1] and not any(overlaps(span, a) for a in argument_spans):
                    candidates.append(span)
        if candidates:
            best = min(candidates, key=lambda p: (span_gap(p, anchor), p))
            break
    return best, source[best[0]:best[1]] != relation


def align_extraction(row, extraction, max_combinations=10000):
    source = normalize_tokens(row["tokens"])
    base = normalize_tokens(tokenize_text(extraction["base_predicate"])[0])
    anchors = find_spans(source, base)
    if not anchors:
        return None, "base_predicate_not_found"
    argument_options = [find_spans(source, normalize_tokens(tokenize_text(a)[0])) for a in extraction["args"]]
    if any(not options for options in argument_options):
        return None, "argument_not_contiguous_in_source"
    combinations = len(anchors) * math.prod(len(x) for x in argument_options)
    if combinations > max_combinations:
        return None, "too_many_ambiguous_alignments"
    choices = []
    for anchor in anchors:
        for arguments in itertools.product(*argument_options):
            spans = (anchor,) + arguments
            if any(overlaps(a, b) for a, b in itertools.combinations(spans, 2)):
                continue
            # Repeated text is disambiguated using proximity, then source order.
            key = (sum(span_gap(anchor, a) for a in arguments), tuple(arguments), anchor)
            choices.append((key, anchor, arguments))
    if not choices:
        return None, "overlapping_roles_in_one_tuple"
    _, anchor, arguments = min(choices, key=lambda item: item[0])
    relation, projected = align_relation(source, normalize_tokens(tokenize_text(extraction["predicate"])[0]), anchor, arguments)
    return {
        "anchor": anchor, "relation": relation, "arguments": arguments,
        "projected_relation": projected, "ambiguous_alignment": len(choices) > 1,
    }, None


def put_bio(labels, span, role):
    begin, end = span
    if any(label != "O" for label in labels[begin:end]):
        raise ValueError("Overlapping roles cannot be encoded in one BIO sequence")
    labels[begin] = f"B-{role}"
    for index in range(begin + 1, end):
        labels[index] = f"I-{role}"


def encode_row(row, tokenizer, max_length):
    full = tokenizer(row["tokens"], is_split_into_words=True, truncation=False, return_attention_mask=True)
    word_ids = full.word_ids()
    count = len(row["tokens"])
    if len(full["input_ids"]) > max_length:
        # BERT adds [CLS] and [SEP]; never retain a partial final WordPiece word.
        last_positions = {w: i for i, w in enumerate(word_ids) if w is not None}
        count = sum(last_positions.get(i, max_length) < max_length - 1 for i in range(count))
        if count == 0:
            raise ValueError("No complete word fits --max-seq-length")
        full = tokenizer(row["tokens"][:count], is_split_into_words=True, truncation=False, return_attention_mask=True)
        word_ids = full.word_ids()
    positions = [word_ids.index(i) for i in range(count)]
    row["features"] = {k: list(full[k]) for k in ("input_ids", "attention_mask", "token_type_ids") if k in full}
    row["features"]["token_type_ids"] = [0] * len(full["input_ids"])
    row["word_ids"] = word_ids
    row["first_positions"] = positions
    row["visible_words"] = count
    return row


def make_example(row, labels=None, label2id=None, anchor=None):
    features = {k: list(v) for k, v in row["features"].items()}
    if anchor is not None:
        features["token_type_ids"] = [int(w is not None and anchor[0] <= w < anchor[1]) for w in row["word_ids"]]
    if labels is not None:
        features["labels"] = [-100] * len(features["input_ids"])
        for word, position in enumerate(row["first_positions"]):
            features["labels"][position] = label2id[labels[word]]
    return {"features": features, "first_positions": row["first_positions"], "anchor": anchor, "row": row}


def prepare_split(rows, tokenizer, max_length, tuple_labels):
    pred_examples, tuple_examples = [], []
    report = Counter(sentences=len(rows), gold_extractions=sum(len(r["extractions"]) for r in rows))
    examples_by_reason = defaultdict(list)
    pred_map = {s: i for i, s in enumerate(PRED_LABELS)}
    tuple_map = {s: i for i, s in enumerate(tuple_labels)}
    for row in rows:
        encode_row(row, tokenizer, max_length)
        if row["visible_words"] < len(row["tokens"]):
            report["truncated_sentences"] += 1
        anchors, uncertain_predicate = set(), False
        seen_tuples, variants = set(), defaultdict(set)
        for extraction in row["extractions"]:
            aligned, reason = align_extraction(row, extraction)
            base_options = find_spans(normalize_tokens(row["tokens"]), normalize_tokens(tokenize_text(extraction["base_predicate"])[0]))
            if aligned:
                anchors.add(aligned["anchor"])
            elif len(base_options) == 1:
                anchors.add(base_options[0])
            else:
                uncertain_predicate = True
            if aligned:
                report["text_aligned_extractions"] += 1
                report["projected_relations"] += int(aligned["projected_relation"])
                report["ambiguous_text_alignments"] += int(aligned["ambiguous_alignment"])
                if f"B-ARG{len(extraction['args']) - 1}" not in tuple_map:
                    reason = "argument_role_unseen_in_training"
                elif max(p[1] for p in [aligned["relation"], *aligned["arguments"]]) > row["visible_words"]:
                    reason = "tuple_crosses_truncation_boundary"
            if reason:
                report[f"excluded_{reason}"] += 1
                if len(examples_by_reason[reason]) < 5:
                    examples_by_reason[reason].append({"sentence": row["source"], "extraction": extraction})
                continue
            report["usable_extractions"] += 1
            labels = ["O"] * len(row["tokens"])
            put_bio(labels, aligned["relation"], "REL")
            for index, span in enumerate(aligned["arguments"]):
                put_bio(labels, span, f"ARG{index}")
            key = (aligned["anchor"], tuple(labels))
            variants[aligned["anchor"]].add(tuple(labels))
            if key in seen_tuples:
                report["duplicate_training_targets"] += 1
                continue
            seen_tuples.add(key)
            tuple_examples.append(make_example(row, labels, tuple_map, aligned["anchor"]))
        row["gold_anchors"] = sorted(anchors)
        report["same_predicate_multiple_gold_variants"] += sum(len(v) > 1 for v in variants.values())
        if any(overlaps(a, b) for a, b in itertools.combinations(anchors, 2)):
            uncertain_predicate = True
        row["predicate_gold_complete"] = not uncertain_predicate and bool(anchors)
        if uncertain_predicate or not anchors:
            report["excluded_predicate_training_sentences"] += 1
            continue
        labels = ["O"] * len(row["tokens"])
        for anchor in sorted(anchors):
            put_bio(labels, anchor, "PRED")
        pred_examples.append(make_example(row, labels, pred_map))
    report["predicate_examples"] = len(pred_examples)
    report["tuple_examples"] = len(tuple_examples)
    report["usable_fraction"] = report["usable_extractions"] / max(1, report["gold_extractions"])
    return pred_examples, tuple_examples, {**dict(report), "excluded_examples": dict(examples_by_reason)}


# ---------------------------------------------------------------------------
# Encoder-only models, word-level BIO decoding, and batched inference
# ---------------------------------------------------------------------------

class BertBIO(torch.nn.Module):
    """One BERT backbone, two fresh BIO classifiers, one joint loss."""

    def __init__(self, bert, tuple_labels):
        super().__init__()
        self.bert = bert
        self.config = bert.config
        dropout = getattr(self.config, "classifier_dropout", None)
        self.dropout = torch.nn.Dropout(
            self.config.hidden_dropout_prob if dropout is None else dropout
        )
        self.predicate_classifier = torch.nn.Linear(self.config.hidden_size, len(PRED_LABELS))
        self.tuple_classifier = torch.nn.Linear(self.config.hidden_size, len(tuple_labels))
        for head in (self.predicate_classifier, self.tuple_classifier):
            torch.nn.init.normal_(head.weight, mean=0.0, std=self.config.initializer_range)
            torch.nn.init.zeros_(head.bias)

    @staticmethod
    def token_loss(logits, labels):
        if labels is None or not torch.any(labels != -100):
            # A mixed batch may contain examples for only one of the heads.
            return logits.sum() * 0.0
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
        )

    def forward(self, input_ids, attention_mask, token_type_ids,
                predicate_labels=None, tuple_labels=None):
        hidden = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state
        hidden = self.dropout(hidden)
        predicate_logits = self.predicate_classifier(hidden)
        tuple_logits = self.tuple_classifier(hidden)
        predicate_loss = self.token_loss(predicate_logits, predicate_labels)
        tuple_loss = self.token_loss(tuple_logits, tuple_labels)
        return SimpleNamespace(
            loss=predicate_loss + tuple_loss,
            predicate_loss=predicate_loss,
            tuple_loss=tuple_loss,
            predicate_logits=predicate_logits,
            tuple_logits=tuple_logits,
        )


def make_model(model_path, labels, tokenizer, seed, gradient_checkpointing=False):
    set_seed(seed)
    config = AutoConfig.from_pretrained(model_path)
    if config.model_type != "bert" or getattr(config, "is_encoder_decoder", False):
        raise ValueError("--model must be an original BERT / PR-BERT checkpoint, not BERT2BERT")
    config.is_decoder = False
    config.add_cross_attention = False
    config.use_cache = False
    config.num_labels = len(labels)
    config.id2label = dict(enumerate(labels))
    config.label2id = {name: index for index, name in enumerate(labels)}
    if config.type_vocab_size < 2:
        raise ValueError("Predicate conditioning requires BERT's two token-type embeddings")
    # Load only the pretrained encoder; old punctuation/LM heads are discarded.
    # No ignore_mismatched_sizes: incompatible backbone weights must fail loudly.
    bert = BertModel.from_pretrained(model_path, config=config, add_pooling_layer=False)
    if len(tokenizer) != bert.get_input_embeddings().num_embeddings:
        raise ValueError("Tokenizer/model vocabulary mismatch; pass the matching --tokenizer")
    torch.manual_seed(seed + 1009)
    model = BertBIO(bert, labels)
    if gradient_checkpointing:
        model.bert.gradient_checkpointing_enable()
    assert_encoder_only(model)
    return model


def assert_encoder_only(model):
    if sum(isinstance(module, BertModel) for module in model.modules()) != 1:
        raise AssertionError("Exactly one shared BERT encoder is required")
    if model.config.is_decoder or model.config.add_cross_attention or model.config.is_encoder_decoder:
        raise AssertionError("BERT must be encoder-only")
    for layer in model.bert.encoder.layer:
        if layer.attention.self.is_decoder or hasattr(layer, "crossattention"):
            raise AssertionError("Unexpected causal attention/cross-attention")


def joint_examples(predicate_examples, tuple_examples):
    """Mix both tasks; every prepared example is visited once per epoch.

    Unmarked sentences supervise only the predicate head. Predicate-marked
    sentences supervise only the tuple head. Both update the same encoder.
    The joint minibatch loss is the sum of the two active mean token losses.
    """
    examples = []
    for task, items in (("predicate", predicate_examples), ("tuple", tuple_examples)):
        for example in items:
            features = {key: list(value) for key, value in example["features"].items()}
            target = features.pop("labels")
            features["predicate_labels"] = [-100] * len(target)
            features["tuple_labels"] = [-100] * len(target)
            features[f"{task}_labels"] = target
            examples.append({**example, "features": features})
    return examples


class BIODataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]["features"]


class BIOCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, items):
        length = max(len(item["input_ids"]) for item in items)
        pads = {"input_ids": self.pad_token_id, "attention_mask": 0, "token_type_ids": 0, "labels": -100,
                "predicate_labels": -100, "tuple_labels": -100}
        return {key: torch.tensor([item[key] + [pads[key]] * (length - len(item[key])) for item in items], dtype=torch.long) for key in items[0]}


def amp_context(device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def bio_decode(log_probs, labels):
    """Viterbi with fixed legality constraints, not a learned CRF."""
    if len(log_probs) == 0:
        return []
    count = len(labels)
    transitions = np.zeros((count, count), dtype=np.float64)
    for new, label in enumerate(labels):
        if label.startswith("I-"):
            role = label[2:]
            for previous, old in enumerate(labels):
                if old not in (f"B-{role}", f"I-{role}"):
                    transitions[previous, new] = -np.inf
    scores = np.array(log_probs[0], dtype=np.float64, copy=True)
    for index, label in enumerate(labels):
        if label.startswith("I-"):
            scores[index] = -np.inf
    backpointers = []
    for emissions in log_probs[1:]:
        candidates = scores[:, None] + transitions
        previous = candidates.argmax(axis=0)
        scores = candidates[previous, np.arange(count)] + emissions
        backpointers.append(previous)
    state = int(scores.argmax())
    path = [state]
    for previous in reversed(backpointers):
        state = int(previous[state])
        path.append(state)
    return list(reversed(path))


def bio_spans(tags):
    spans, start, role = [], None, None
    for index, tag in enumerate(list(tags) + ["O"]):
        prefix, current = tag.split("-", 1) if "-" in tag else ("O", None)
        continuation = prefix == "I" and current == role and role is not None
        if not continuation and role is not None:
            spans.append((role, start, index))
            start, role = None, None
        if prefix in ("B", "I") and not continuation:
            start, role = index, current
    return spans


@torch.no_grad()
def predict_words(model, examples, labels, args, device, task):
    model.eval()
    output = []
    collate = BIOCollator(model.config.pad_token_id)
    for begin in range(0, len(examples), args.eval_batch_size):
        chunk = examples[begin:begin + args.eval_batch_size]
        features = [{k: v for k, v in ex["features"].items() if k not in ("labels", "predicate_labels", "tuple_labels")} for ex in chunk]
        batch = {k: v.to(device) for k, v in collate(features).items()}
        with amp_context(device):
            logits = getattr(model(**batch), f"{task}_logits")
        log_probs = logits.float().log_softmax(-1).cpu().numpy()
        for example, probabilities in zip(chunk, log_probs):
            word_probs = probabilities[example["first_positions"]]
            constrained = word_probs.copy()
            anchor = example["anchor"]
            if anchor is not None:
                # A selected predicted anchor MUST remain part of its relation.
                allowed = {labels.index("B-REL"), labels.index("I-REL")}
                forbidden = [i for i in range(len(labels)) if i not in allowed]
                constrained[anchor[0]:anchor[1], forbidden] = -np.inf
            ids = bio_decode(constrained, labels)
            output.append(([labels[i] for i in ids], [float(word_probs[w, i]) for w, i in enumerate(ids)]))
    return output


def detect_predicates(model, rows, args, device):
    examples = [make_example(row) for row in rows]
    outputs = predict_words(model, examples, PRED_LABELS, args, device, task="predicate")
    predictions = []
    for tags, log_probs in outputs:
        predicates = []
        for role, begin, end in bio_spans(tags):
            score = math.exp(float(np.mean(log_probs[begin:end])))
            predicates.append({"span": (begin, end), "confidence": score})
        predictions.append(predicates)
    return predictions


def predicate_f1(predictions, rows):
    correct = attempted = gold_count = 0
    for predictions_here, row in zip(predictions, rows):
        if not row.get("predicate_gold_complete", True):
            # This affects only the predicate span diagnostic.
            # Official OIE dev and test evaluation NEVER filters these rows.
            continue
        found = {tuple(p["span"]) for p in predictions_here}
        gold = set(row["gold_anchors"])
        correct += len(found & gold)
        attempted += len(found)
        gold_count += len(gold)
    precision = correct / attempted if attempted else 0.0
    recall = correct / gold_count if gold_count else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def span_text(row, begin, end):
    return row["source"][row["offsets"][begin][0]:row["offsets"][end - 1][1]]


def tuple_from_tags(row, tags, log_probs, predicate_confidence):
    roles = defaultdict(list)
    for role, begin, end in bio_spans(tags):
        roles[role].append(span_text(row, begin, end))
    if not roles["REL"] or not roles["ARG0"]:
        return None
    numbered = sorted((int(role[3:]), " ".join(parts)) for role, parts in roles.items() if role.startswith("ARG") and parts)
    selected = [log_probs[i] for i, tag in enumerate(tags) if tag != "O"]
    confidence = predicate_confidence * math.exp(float(np.mean(selected)))
    return {"predicate": " ".join(roles["REL"]), "args": [text for _, text in numbered], "confidence": confidence, "roles": {f"ARG{i}": text for i, text in numbered}}


def extract_tuples(model, rows, predicate_predictions, labels, args, device):
    examples, owners = [], []
    results = [{"text": row["source"], "predicted_predicates": pred, "extractions": []} for row, pred in zip(rows, predicate_predictions)]
    for index, (row, predicates) in enumerate(zip(rows, predicate_predictions)):
        for pred in predicates:
            examples.append(make_example(row, anchor=pred["span"]))
            owners.append((index, pred))
    outputs = predict_words(model, examples, labels, args, device, task="tuple")
    for (index, pred), (tags, probabilities) in zip(owners, outputs):
        extraction = tuple_from_tags(rows[index], tags, probabilities, pred["confidence"])
        if extraction:
            extraction["anchor_word_span"] = pred["span"]
            extraction["bio_tags"] = tags
            results[index]["extractions"].append(extraction)
    for result in results:
        unique = {}
        for extraction in result["extractions"]:
            key = (extraction["predicate"], tuple(extraction["args"]))
            if key not in unique or extraction["confidence"] > unique[key]["confidence"]:
                unique[key] = extraction
        result["extractions"] = list(unique.values())
    return results


# Benchmark

def prepare_benchmark(repo_dir):
    import sklearn.preprocessing
    if "sklearn.preprocessing.data" not in sys.modules:
        shim = types.ModuleType("sklearn.preprocessing.data")
        shim.binarize = sklearn.preprocessing.binarize
        sys.modules["sklearn.preprocessing.data"] = shim
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    for name, probe in [("stopwords", lambda: stopwords.words("english")), ("wordnet", lambda: WordNetLemmatizer().lemmatize("tests"))]:
        try:
            probe()
        except LookupError as exc:
            raise RuntimeError(
                f"Missing NLTK {name}. Install before submitting: "
                "python -m nltk.downloader stopwords wordnet"
            ) from exc
    repo = str(Path(repo_dir).resolve())
    if "benchmark" in sys.modules:
        loaded = Path(sys.modules["benchmark"].__file__).resolve().parent
        if str(loaded) != repo:
            raise RuntimeError(f"A different benchmark module is already imported: {loaded}")
    sys.path.insert(0, repo)
    from benchmark import Benchmark
    from matcher import Matcher
    from oie_readers.extraction import Extraction
    return Benchmark, Matcher, Extraction


def summarize_pr_curve(precision, recall):
    precision, recall = np.asarray(precision), np.asarray(recall)
    if not len(precision):
        return dict(best_precision=0.0, best_recall=0.0, best_f1=0.0, pr_auc=0.0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator > 0)
    best = int(np.argmax(f1))
    order = np.argsort(recall)  # Same AUC calculation as the old OIE runner.
    return {"best_precision": float(precision[best]), "best_recall": float(recall[best]), "best_f1": float(f1[best]), "pr_auc": float(auc(recall[order], precision[order])) if len(precision) > 1 else 0.0}


def official_evaluate(predictions, gold_path, benchmark_api, curve_path):
    Benchmark, Matcher, Extraction = benchmark_api
    predicted = defaultdict(list)
    count = 0
    for row in predictions:
        for data in row["extractions"]:
            extraction = Extraction(pred=data["predicate"], sent=row["text"], confidence=float(data["confidence"]))
            for arg in data["args"]:
                extraction.addArg(arg)
            predicted[row["text"]].append(extraction)
            count += 1
    curve_path = Path(curve_path)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    if count == 0:
        curve_path.write_text("Precision\tRecall\n0.0\t0.0\n", encoding="utf-8")
        metrics = summarize_pr_curve([0.0], [0.0])
    else:
        benchmark = Benchmark(str(gold_path))
        benchmark.compare(predicted=predicted, matchingFunc=Matcher.lexicalMatch, output_fn=str(curve_path))
        precision, recall = [], []
        with curve_path.open(encoding="utf-8") as handle:
            next(handle, None)
            for line in handle:
                if line.strip():
                    p, r = line.split()
                    precision.append(float(p))
                    recall.append(float(r))
        metrics = summarize_pr_curve(precision, recall)
    metrics["num_predicted_extractions"] = count
    metrics["num_parsed_predictions"] = count  # compatible with old result naming
    return metrics


# ---------------------------------------------------------------------------
# Joint training and results:
# ---------------------------------------------------------------------------

def train_joint(model, predicate_examples, tuple_examples, tokenizer, args, device, evaluate):
    if not predicate_examples or not tuple_examples:
        raise ValueError("Both tasks need usable training examples; inspect alignment diagnostics")
    examples = joint_examples(predicate_examples, tuple_examples)
    model.to(device)
    loader = DataLoader(
        BIODataset(examples), batch_size=args.train_batch_size, shuffle=True,
        collate_fn=BIOCollator(tokenizer.pad_token_id), num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed), worker_init_fn=seed_worker,
        pin_memory=device.type == "cuda",
    )
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if name.endswith("bias") or "LayerNorm.weight" in name else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.learning_rate, eps=1e-8,
    )
    total_steps = math.ceil(len(loader) / args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = Counter()
        for step, batch in enumerate(tqdm(loader, desc=f"Joint BIO epoch {epoch}/{args.epochs}")):
            batch = {key: value.to(device) for key, value in batch.items()}
            group_start = (step // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, len(loader) - group_start)
            with amp_context(device):
                output = model(**batch)
            if not torch.isfinite(output.loss):
                raise FloatingPointError(f"Non-finite joint loss at epoch {epoch}, step {step}")
            (output.loss / group_size).backward()
            for name in ("loss", "predicate_loss", "tuple_loss"):
                losses[name] += float(getattr(output, name).detach())
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        # Recompute predicates every epoch: the shared encoder has changed.
        details = evaluate(model, epoch)
        record = {"epoch": epoch, **{f"train_{k}": v / len(loader) for k, v in losses.items()}, **details}
        history.append(record)
        print(f"[joint BIO] {json.dumps(record)}", flush=True)
    assert_encoder_only(model)
    return {"evaluated_epoch": args.epochs, "selection": "final_epoch", "epochs": history}


def append_metrics_csv(path, row):
    """Append one run under a Linux file lock; write the header only once."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row)
    with path.open("a+", newline="", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        header = next(csv.reader(handle), None)
        if header is not None and header != fields:
            raise ValueError(f"CSV columns do not match the existing file: {path}")
        handle.seek(0, 2)
        writer = csv.DictWriter(handle, fieldnames=fields)
        if header is None:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_bert(args, model_path, tokenizer, splits, examples, audit, benchmark_api):
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    prefix = args.run_name or f"oie-bert-bio-{safe_name(args.model)}-{args.epochs}ep"
    if Path(prefix).name != prefix:
        raise ValueError("--run-name must be a filename prefix, not a path")
    run_name = f"{prefix}-seed{args.seed}"
    paths = {suffix: Path(args.results_dir) / f"{run_name}.{suffix}" for suffix in
             ("alignment.json", "metrics.json", "predictions.jsonl", "pr.dat", "dev.pr.dat")}
    for path in paths.values():
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Will not overwrite {path}; use a new --run-name or --overwrite")
    # The shared CSV is deliberately excluded from per-run overwrite checks.
    paths["metrics.csv"] = Path(args.results_dir) / "oie_bert_BIO_tags_results.csv"
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    paths["alignment.json"].write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("\n============================================================")
    print(f"Architecture:       one BERT encoder, two BIO heads")
    print(f"Model source:       {args.model}")
    print(f"Resolved model:     {model_path}")
    print(f"Seed:               {args.seed}")
    print(f"Epochs:             {args.epochs} (final epoch evaluated)")
    print(f"Learning rate:      {args.learning_rate}")
    print(f"Device:             {device}")
    print("Evaluation:         Benchmark + Matcher.lexicalMatch")
    print("Test predicates:    predicted, never supplied from gold")
    print("Checkpoint saving:  disabled")
    print("============================================================\n")

    model = make_model(model_path, BIO_LABELS, tokenizer, args.seed, args.gradient_checkpointing)

    def evaluate_dev(current_model, epoch):
        predicates = detect_predicates(current_model, splits["dev"], args, device)
        predictions = extract_tuples(current_model, splits["dev"], predicates, BIO_LABELS, args, device)
        metrics = official_evaluate(
            predictions, Path(args.data_dir) / "dev.oie", benchmark_api, paths["dev.pr.dat"]
        )
        pred_scores = predicate_f1(predicates, splits["dev"])
        return {**{"dev_" + key: value for key, value in metrics.items()},
                **{"dev_predicate_" + key: value for key, value in pred_scores.items()}}

    history = train_joint(
        model, examples["train"][0], examples["train"][1],
        tokenizer, args, device, evaluate_dev,
    )
    # Test is used only after all training; all test sentences are retained.
    predicates = detect_predicates(model, splits["test"], args, device)
    predictions = extract_tuples(model, splits["test"], predicates, BIO_LABELS, args, device)
    metrics = official_evaluate(
        predictions, Path(args.data_dir) / "test.oie", benchmark_api, paths["pr.dat"]
    )
    with paths["predictions.jsonl"].open("w", encoding="utf-8") as handle:
        for row, prediction in zip(splits["test"], predictions):
            record = {**prediction, "gold_extractions": row["extractions"], "tokens": row["tokens"],
                      "visible_words": row["visible_words"]}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metrics.update({
        "run_name": run_name, "architecture": "bert_encoder_only_shared_bio",
        "num_bert_encoders": 1, "num_classification_heads": 2,
        "is_decoder": False, "add_cross_attention": False, "seed": args.seed,
        "model_source": args.model, "resolved_model_path": model_path,
        "tokenizer": args.tokenizer, "configuration": vars(args),
        "predicate_labels": PRED_LABELS, "tuple_labels": BIO_LABELS,
        "training": history, "matching_function": "Matcher.lexicalMatch",
        "gold_test_file": str(Path(args.data_dir) / "test.oie"),
        "predicates_at_test": "predicted_not_gold", "gold_test_filtered": False,
        "alignment": audit, "files": {key: str(value) for key, value in paths.items()},
        "torch_version": torch.__version__, "transformers_version": transformers_version,
    })
    paths["metrics.json"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Baseline and PR seeds append to the same CSV; existing rows are retained.
    csv_row = {
        "run_name": run_name,
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "precision": metrics["best_precision"],
        "recall": metrics["best_recall"],
        "f1": metrics["best_f1"],
        "pr_auc": metrics["pr_auc"],
    }
    append_metrics_csv(paths["metrics.csv"], csv_row)
    print("\n================ OIE2016 OFFICIAL TEST =================")
    print(f"Run:             {run_name}")
    print(f"Best Precision:  {metrics['best_precision']:.6f}")
    print(f"Best Recall:     {metrics['best_recall']:.6f}")
    print(f"Best F1:         {metrics['best_f1']:.6f}")
    print(f"PR AUC:          {metrics['pr_auc']:.6f}")
    print(f"Predictions:     {paths['predictions.jsonl']}")
    print(f"Metrics:         {paths['metrics.json']}")
    print(f"CSV:             {paths['metrics.csv']}")
    print("========================================================\n")
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


# ---------------------------------------------------------------------------
# CLI: model supplied through -m/--model or as a positional argument
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fine-tune one shared BERT encoder with BIO heads on OIE2016.",
    )
    parser.add_argument("model_positional", nargs="?", metavar="MODEL",
                        help="BERT or PR-BERT checkpoint, as an alternative to -m.")
    parser.add_argument("-m", "--model", help="BERT or PR-BERT HF checkpoint.")
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--benchmark-repo", help="OIE parent folder or the oie-benchmark clone.")
    parser.add_argument("--data-dir")
    parser.add_argument("--results-dir", default="outputs/oie_results")
    parser.add_argument("--run-name", help="Filename prefix; -seedN is appended.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-e", "--epochs", type=int, default=5)
    parser.add_argument("--train-batch-size", "--batch", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", "--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--min-alignment-coverage", type=float, default=0.5)
    parser.add_argument("--check-data", action="store_true", help="Audit data and exit without training.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.model is not None and args.model_positional is not None:
        parser.error("Supply the model either with -m/--model or positionally, not both.")
    args.model = args.model if args.model is not None else args.model_positional
    if not args.model:
        parser.error("A model is required: use -m MODEL or supply MODEL positionally.")
    del args.model_positional
    for name in ("epochs", "train_batch_size", "eval_batch_size", "grad_accum"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 4 <= args.max_seq_length <= 512:
        parser.error("--max-seq-length must be between 4 and 512 for BERT")
    if args.num_workers < 0 or args.learning_rate <= 0 or args.max_grad_norm <= 0 or args.weight_decay < 0:
        parser.error("Invalid workers, learning rate, max grad norm, or weight decay")
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.min_alignment_coverage <= 1:
        parser.error("Invalid warmup ratio or minimum alignment coverage")
    if not 0 <= args.seed < 2**32 - 1009:
        parser.error("Seed must be a nonnegative integer below 2**32 - 1009")
    return args


def main(argv=None):
    args = parse_args(argv)
    resolve_repo_layout(args)
    model_path = resolve_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if not tokenizer.is_fast or tokenizer.pad_token_id is None:
        raise ValueError("A fast BERT tokenizer with a padding token is required")
    config = AutoConfig.from_pretrained(model_path)
    if config.model_type != "bert" or config.is_encoder_decoder:
        raise ValueError("Supply an original BERT / PR-BERT checkpoint, not BERT2BERT")
    if args.max_seq_length > config.max_position_embeddings:
        raise ValueError("--max-seq-length exceeds this checkpoint's positional embeddings")
    splits = {
        split: read_oie_split(Path(args.data_dir) / f"{split}.oie")
        for split in ("train", "dev", "test")
    }
    # Fail rather than silently discard train roles outside the declared schema.
    max_args = max(len(ex["args"]) for row in splits["train"] for ex in row["extractions"])
    if f"B-ARG{max_args - 1}" not in BIO_LABELS:
        raise ValueError(f"Training has {max_args} arguments; extend BIO_LABELS at the top of this file.")
    examples, audit = {}, {}
    for split, rows in splits.items():
        pred, tuples, report = prepare_split(rows, tokenizer, args.max_seq_length, BIO_LABELS)
        examples[split] = (pred, tuples)
        audit[split] = report
        print(f"[alignment {split}] {json.dumps({k: v for k, v in report.items() if k != 'excluded_examples'})}", flush=True)
    audit["exact_sentence_overlap"] = {
        f"{a}/{b}": len({r["source"] for r in splits[a]} & {r["source"] for r in splits[b]})
        for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    if any(audit["exact_sentence_overlap"].values()):
        print(f"WARNING: exact-sentence overlaps in supplied splits: {audit['exact_sentence_overlap']}; splits NOT modified")
    if args.check_data:
        print(json.dumps(audit, indent=2))
        print("Data audit complete. No training or test scoring performed.")
        return
    # Preflight the scorer before spending time loading and training a model.
    benchmark_api = prepare_benchmark(args.benchmark_repo)
    print("Official oie-benchmark evaluator preflight: PASS")
    if audit["train"]["usable_fraction"] < args.min_alignment_coverage:
        raise ValueError("Training alignment coverage is below --min-alignment-coverage; inspect --check-data.")
    run_bert(args, model_path, tokenizer, splits, examples, audit, benchmark_api)


if __name__ == "__main__":
    main()
