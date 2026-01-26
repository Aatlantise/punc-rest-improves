import re
import os
import torch
import numpy as np
from collections import defaultdict
from datasets import load_from_disk
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

# --- Configuration ---
MODEL_CHECKPOINT = "t5-base"
# Update this path if you have the dataset locally or use the hub ID
DATASET_NAME = "data/finance-news"
MAX_SEQ_LENGTH = 512
OVERLAP = 100
STRIDE = MAX_SEQ_LENGTH - OVERLAP
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
EPOCHS = 30
OUTPUT_DIR = "./pr-t5-financial-news"
LOGGING_DIR = "./logs"

# Punctuation set for evaluation
TARGET_PUNCT_SET = {'.', ',', '!', '?', ';', ':', '"', "'"}


# --- Preprocessing Helpers ---

def clean_text(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def corruption_function(text):
    # Remove: . , ! ? : ; " '
    punctuation_regex = r'[.,!?;:"\']'
    depunctuated = re.sub(punctuation_regex, '', text)
    return depunctuated.lower()


def sliding_window_preprocess(examples, tokenizer):
    model_inputs = {"input_ids": [], "labels": []}

    # 1. Calculate the cost of the prefix and special tokens
    # T5 typically adds 1 EOS token. We verify the prefix length.
    prefix = "restore punctuation: "
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    prefix_len = len(prefix_ids)

    # Reserve 1 token for EOS </s>
    # Safe content length = 512 - prefix (approx 4) - EOS (1) = ~507
    safe_chunk_size = MAX_SEQ_LENGTH - prefix_len - 1

    for text in examples['text']:
        cleaned_text = clean_text(text)
        if len(cleaned_text) < 10:
            continue

        full_token_ids = tokenizer(cleaned_text, add_special_tokens=False)["input_ids"]

        # 2. Use SAFE_CHUNK_SIZE instead of MAX_SEQ_LENGTH for the slice
        for i in range(0, len(full_token_ids), STRIDE):
            chunk_ids = full_token_ids[i: i + safe_chunk_size]

            if len(chunk_ids) < 20:
                continue

            target_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            source_text = prefix + corruption_function(target_text)

            # 3. Now truncation will only hit if something goes wrong, not by default
            tokenized_input = tokenizer(source_text, max_length=MAX_SEQ_LENGTH, truncation=True)
            tokenized_target = tokenizer(target_text, max_length=MAX_SEQ_LENGTH, truncation=True)

            # Sanity Check: Ensure EOS token is present
            # T5's EOS token ID is usually 1.
            if tokenized_target["input_ids"][-1] != tokenizer.eos_token_id:
                # If needed, manually append, though tokenizer usually handles this if space permits
                pass

            model_inputs["input_ids"].append(tokenized_input["input_ids"])
            model_inputs["labels"].append(tokenized_target["input_ids"])

    return model_inputs


# --- Evaluation Helpers ---

def extract_word_and_punct(token):
    if not token:
        return "", ""
    last_char = token[-1]
    if last_char in TARGET_PUNCT_SET:
        return token[:-1], last_char
    return token, ""


def get_stats_for_pair(target_text, pred_text):
    """
    Aligns and calculates TP/FP/FN for a single sentence pair.
    """

    target_tokens = target_text.split()
    pred_tokens = pred_text.split()

    stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    aligned_words = 0

    t_idx, p_idx = 0, 0

    while t_idx < len(target_tokens) and p_idx < len(pred_tokens):
        t_word, t_punct = extract_word_and_punct(target_tokens[t_idx])
        p_word, p_punct = extract_word_and_punct(pred_tokens[p_idx])

        if t_word.lower() == p_word.lower():
            aligned_words += 1
            if t_punct == p_punct:
                if t_punct != "":
                    stats[t_punct]["TP"] += 1
            else:
                if t_punct != "": stats[t_punct]["FN"] += 1
                if p_punct != "": stats[p_punct]["FP"] += 1
            t_idx += 1
            p_idx += 1
        else:
            p_idx += 1

    return stats, aligned_words


def compute_metrics(eval_preds, tokenizer):
    """
    The function called by Trainer during evaluation.
    """
    preds, labels = eval_preds

    # Replace -100 in the labels as we can't decode them
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in labels with pad_token_id (usually 0 for T5) to decode correctly
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    for i in range(3):
        print(f"Predicted: {decoded_preds[i]}")
        print(f"Target: {decoded_labels[i]}")

    # Aggregators
    total_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    total_aligned_words = 0

    # Iterate over batch
    for pred_text, target_text in zip(decoded_preds, decoded_labels):
        batch_stats, batch_words = get_stats_for_pair(target_text, pred_text)
        total_aligned_words += batch_words
        for punct, counts in batch_stats.items():
            total_stats[punct]["TP"] += counts["TP"]
            total_stats[punct]["FP"] += counts["FP"]
            total_stats[punct]["FN"] += counts["FN"]

    # Calculate Scores
    results = {}
    macro_f1_scores = []

    for punct in sorted(TARGET_PUNCT_SET):
        tp = total_stats[punct]["TP"]
        fp = total_stats[punct]["FP"]
        fn = total_stats[punct]["FN"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        results[f"f1_{punct}"] = f1
        macro_f1_scores.append(f1)

    results["macro_f1"] = sum(macro_f1_scores) / len(macro_f1_scores) if macro_f1_scores else 0.0
    return results


def main(debug=False):
    # 1. Setup
    tokenizer = T5Tokenizer.from_pretrained(MODEL_CHECKPOINT, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT)

    # 2. Load and Process Dataset (using a subset for demo speed, remove slice for full run)
    print("Loading and processing data...")
    if debug:
        dataset = load_from_disk(DATASET_NAME).select(range(1000))
        EPOCHS = 3
    else:
        dataset = load_from_disk(DATASET_NAME)

    processed_dataset = dataset.map(
        lambda x: sliding_window_preprocess(x, tokenizer),
        batched=True,
        batch_size=100,
        remove_columns=dataset.column_names,
        num_proc=4
    )

    # Split: Train vs Eval
    split = processed_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # 3. Training Args
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",  # Evaluate every epoch
        save_strategy="epoch",  # Save every epoch
        logging_strategy="steps",
        logging_steps=500,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_checkpointing=True,
        weight_decay=0.01,
        num_train_epochs=EPOCHS,
        predict_with_generate=True,  # Essential for compute_metrics to receive predictions
        fp16=False,
        bf16=True,
        report_to="tensorboard",
        logging_dir=LOGGING_DIR,
    )

    # 4. Initialize Trainer with compute_metrics
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        tokenizer=tokenizer,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer)
    )

    # 5. Train
    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()