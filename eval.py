import torch
import string
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_from_disk
from tqdm import tqdm

# --- Configuration ---
MODEL_PATH = "/home/jm3743/pr-t5/pr-t5"
TEST_DATA_PATH = "/home/jm3743/pr-t5/data/finance-news/test"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_word_and_punct(token):
    """
    Splits a token (e.g., "earnings.") into ("earnings", ".")
    Returns (word, punctuation_mark)
    """
    # Check if the last character is punctuation
    if token and token[-1] in string.punctuation:
        return token[:-1], token[-1]
    return token, ""


def align_and_score(target_text, pred_text):
    """
    Aligns orthographic words and calculates TP, FP, TN, FN.
    Assumption: The model performs restoration, so the word sequence
    (ignoring case and punctuation) should be nearly identical.
    """
    target_tokens = target_text.split()
    pred_tokens = pred_text.split()

    stats = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    # Simple alignment: We iterate through both lists.
    # Since T5 might hallucinate or skip words slightly, we use a pointer approach.
    # However, for Punctuation Restoration specifically, usually lengths match closely.

    t_idx = 0
    p_idx = 0

    while t_idx < len(target_tokens) and p_idx < len(pred_tokens):
        t_word, t_punct = extract_word_and_punct(target_tokens[t_idx])
        p_word, p_punct = extract_word_and_punct(pred_tokens[p_idx])

        # Normalize for comparison
        if t_word.lower() == p_word.lower():
            # Words match, check punctuation

            # Case 1: Target has punctuation
            if t_punct:
                if p_punct == t_punct:
                    stats["TP"] += 1  # Correctly predicted missing punctuation
                elif p_punct == "":
                    stats["FN"] += 1  # Incorrectly predicted NO punctuation
                else:
                    # Target has '.', Pred has ',' -> This is technically a wrong label.
                    # Based on your definitions, if we strictly care about "restoring",
                    # we can count this as a generic failure, or map it.
                    # Here we treat "wrong punctuation" as a failure to restore the *correct* one.
                    # It acts like a FN for the correct class.
                    stats["FN"] += 1

                    # Case 2: Target has NO punctuation
            else:
                if p_punct:
                    stats["FP"] += 1  # Incorrectly predicted punctuation
                else:
                    stats["TN"] += 1  # Correctly predicted no punctuation

            t_idx += 1
            p_idx += 1
        else:
            # Words do not match (Misalignment/Hallucination)
            # Advance the pointer of the token that is "behind" or simply skip
            # For simplicity in this script, we skip mismatching words to stay aligned
            # In a robust production system, you'd use Needleman-Wunsch alignment.
            p_idx += 1
            # (Heuristic: assume prediction inserted an extra word or got wrong word)

    return stats


def main():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)

    print(f"Loading test data from {TEST_DATA_PATH}...")
    dataset = load_from_disk(TEST_DATA_PATH)

    # Metrics Aggregator
    total_stats = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    print("Running evaluation...")
    model.eval()

    # Process a subset or full set
    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        input_ids = torch.tensor([example["input_ids"]]).to(DEVICE)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )

        # Decode
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        target_text = tokenizer.decode(example["labels"], skip_special_tokens=True)

        # Calculate Stats
        batch_stats = align_and_score(target_text, pred_text)

        for k in total_stats:
            total_stats[k] += batch_stats[k]

    # Calculate Final Metrics
    tp = total_stats["TP"]
    fp = total_stats["FP"]
    fn = total_stats["FN"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- Evaluation Results ---")
    print(f"True Positives  (Correct Punct): {tp}")
    print(f"False Positives (Hallucinated) : {fp}")
    print(f"True Negatives  (Correct Blank): {total_stats['TN']}")
    print(f"False Negatives (Missed Punct) : {fn}")
    print("-" * 30)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")


if __name__ == "__main__":
    main()