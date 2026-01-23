import torch
import string
import re
from collections import defaultdict
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_from_disk
from tqdm import tqdm

# --- Configuration ---
MODEL_PATH = "/home/jm3743/pr-t5/pr-t5"
TEST_DATA_PATH = "/home/jm3743/pr-t5/data/finance-news/test"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The specific punctuation marks we want to evaluate
TARGET_PUNCT_SET = {'.', ',', '!', '?', ';', ':', '"', "'"}


def extract_word_and_punct(token):
    """
    Splits a token (e.g., "earnings.") into ("earnings", ".")
    Returns (word, punctuation_mark).
    Only considers punctuation if it is in TARGET_PUNCT_SET.
    """
    if not token:
        return "", ""

    last_char = token[-1]

    # Only separate if it's a punctuation mark we care about
    if last_char in TARGET_PUNCT_SET:
        return token[:-1], last_char

    # If it's some other punctuation (like $) or just a letter, keep it as word
    return token, ""


def align_and_score(target_text, pred_text):
    """
    Aligns words and calculates TP, FP, FN for each punctuation class.
    Also counts 'correct_no_punct' to calculate TN later.
    """
    target_tokens = target_text.split()
    pred_tokens = pred_text.split()

    # Structure: stats[punct_char] = {'TP': 0, 'FP': 0, 'FN': 0}
    # We will calculate TN dynamically based on total words processed.
    batch_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

    t_idx = 0
    p_idx = 0
    aligned_word_count = 0

    while t_idx < len(target_tokens) and p_idx < len(pred_tokens):
        t_word, t_punct = extract_word_and_punct(target_tokens[t_idx])
        p_word, p_punct = extract_word_and_punct(pred_tokens[p_idx])

        # Normalize word text for alignment (ignore case)
        if t_word.lower() == p_word.lower():
            aligned_word_count += 1

            # --- Logic for Specific Punctuation Classes ---

            if t_punct == p_punct:
                if t_punct != "":
                    # Case: Both have same punctuation (e.g., Target='.', Pred='.')
                    batch_stats[t_punct]["TP"] += 1
                else:
                    # Case: Both have NO punctuation
                    # This contributes to TN for ALL punctuation classes
                    pass
            else:
                # Mismatch Cases

                # 1. Handle Target Punctuation (The Missed One)
                if t_punct != "":
                    # Target has punct, Pred does not (or has wrong one) -> FN for t_punct
                    batch_stats[t_punct]["FN"] += 1

                # 2. Handle Predicted Punctuation (The Hallucinated One)
                if p_punct != "":
                    # Pred has punct, Target does not (or has wrong one) -> FP for p_punct
                    batch_stats[p_punct]["FP"] += 1

            t_idx += 1
            p_idx += 1
        else:
            # Simple skip on mismatch (assume extra/missing word in pred)
            # You might want to advance the index that is 'behind' lexicographically
            # or just skip the prediction index.
            p_idx += 1

    return batch_stats, aligned_word_count


def calculate_metrics(stats, total_words):
    """
    Computes Precision, Recall, F1, and TN for each class, plus Micro/Macro averages.
    """
    results = {}

    # -- 1. Per-Class Metrics --
    sum_tp = 0
    sum_fp = 0
    sum_fn = 0

    macro_f1_scores = []

    for punct in sorted(TARGET_PUNCT_SET):
        tp = stats[punct]["TP"]
        fp = stats[punct]["FP"]
        fn = stats[punct]["FN"]

        # TN = (Total Aligned Words) - (TP + FP + FN)
        # Note: This definition of TN is "Times this specific punct was correctly NOT predicted"
        # It excludes times where we predicted it (FP), missed it (FN), or correctly predicted it (TP).
        tn = total_words - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        results[punct] = {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": precision, "Recall": recall, "F1": f1
        }

        # Accumulate for Micro/Macro
        sum_tp += tp
        sum_fp += fp
        sum_fn += fn
        macro_f1_scores.append(f1)

    # -- 2. Micro Average --
    # Micro treats all instances equally
    micro_prec = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
    micro_rec = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    # -- 3. Macro Average --
    # Macro averages the scores of the classes (treating all classes equally)
    macro_f1 = sum(macro_f1_scores) / len(macro_f1_scores) if macro_f1_scores else 0.0

    return results, micro_f1, macro_f1


def main():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)

    print(f"Loading test data from {TEST_DATA_PATH}...")
    dataset = load_from_disk(TEST_DATA_PATH)

    # Global Aggregators
    total_class_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    total_aligned_words = 0

    print("Running evaluation...")
    model.eval()

    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        input_ids = torch.tensor([example["input_ids"]]).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )

        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        target_text = tokenizer.decode(example["labels"], skip_special_tokens=True)

        # Get batch stats
        batch_stats, batch_word_count = align_and_score(target_text, pred_text)

        # Aggregate
        total_aligned_words += batch_word_count
        for punct, counts in batch_stats.items():
            total_class_stats[punct]["TP"] += counts["TP"]
            total_class_stats[punct]["FP"] += counts["FP"]
            total_class_stats[punct]["FN"] += counts["FN"]

    # Calculate final metrics
    per_class_results, micro_f1, macro_f1 = calculate_metrics(total_class_stats, total_aligned_words)

    # --- Print Report ---
    print("\n" + "=" * 80)
    print(f"{'Punctuation':<12} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'TN':<8} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
    print("-" * 80)

    for punct, metrics in per_class_results.items():
        # Handle ' (single quote) display
        display_punct = punct if punct != "'" else "\\'"
        print(
            f"{display_punct:<12} | {metrics['TP']:<6} | {metrics['FP']:<6} | {metrics['FN']:<6} | {metrics['TN']:<8} | {metrics['Precision']:.3f}  | {metrics['Recall']:.3f}  | {metrics['F1']:.3f}")

    print("-" * 80)
    print(f"OVERALL MICRO F1: {micro_f1:.4f}")
    print(f"OVERALL MACRO F1: {macro_f1:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()