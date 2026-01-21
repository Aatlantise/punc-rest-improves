import re
import torch
import ftfy
from datasets import load_dataset, load_from_disk
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

# --- Configuration ---
MODEL_CHECKPOINT = "/home/jm3743/pr-t5/t5-base"
DATASET_DIR = "/home/jm3743/pr-t5/data/finance-news"
MAX_SEQ_LENGTH = 512
OVERLAP = 100
STRIDE = MAX_SEQ_LENGTH - OVERLAP  # How much we advance the window
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
EPOCHS = 3
OUTPUT_DIR = "/home/jm3743/pr-t5/pr-t5"
LOGGING_DIR = "/home/jm3743/pr-t5/logs"

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # 1. Fix encoding issues and normalize quotes/dashes (e.g., ’ -> ')
    text = ftfy.fix_text(text)

    # 2. Remove excessive whitespace
    return " ".join(text.split())


def corruption_function(text):
    """
    Removes punctuation and lowercases text to create the source input.
    """
    # Remove: . , ! ? : ; " '
    punctuation_regex = r'[.,!?;:"\']'
    depunctuated = re.sub(punctuation_regex, '', text)
    return depunctuated.lower()


def sliding_window_preprocess(examples, tokenizer):
    """
    1. Tokenize the raw long text.
    2. Slide a window of size 512 with overlap 100 over the tokens.
    3. Decode back to text (Target).
    4. Corrupt text (Source).
    5. Re-tokenize for model input.
    """
    model_inputs = {"input_ids": [], "labels": []}

    # Iterate over the raw examples in the batch
    for text in examples['text']:
        cleaned_text = clean_text(text)
        if len(cleaned_text) < 10:
            continue

        # 1. Tokenize the full text first (without truncation)
        # We need the IDs to perform the precise sliding window logic
        full_token_ids = tokenizer(cleaned_text, add_special_tokens=False)["input_ids"]

        # 2. Create chunks
        for i in range(0, len(full_token_ids), STRIDE):
            chunk_ids = full_token_ids[i: i + MAX_SEQ_LENGTH]

            # Skip very small chunks at the end
            if len(chunk_ids) < 20:
                continue

            # 3. Decode chunk to get the "Target" string
            # We decode so we can apply regex-based corruption reliably on text
            target_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)

            # 4. Create "Source" string
            source_text = corruption_function(target_text)
            source_text = "restore punctuation: " + source_text

            # 5. Re-tokenize inputs and targets
            tokenized_input = tokenizer(
                source_text,
                max_length=MAX_SEQ_LENGTH,
                truncation=True
            )
            tokenized_target = tokenizer(
                target_text,
                max_length=MAX_SEQ_LENGTH,
                truncation=True
            )

            model_inputs["input_ids"].append(tokenized_input["input_ids"])
            model_inputs["labels"].append(tokenized_target["input_ids"])

    return model_inputs


def main():
    # 1. Setup
    tokenizer = T5Tokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT)

    # 2. Load Dataset
    # Using 'train' split initially, we will split manually later
    raw_dataset = load_from_disk(DATASET_DIR)

    # 3. Apply Sliding Window & Preprocessing
    print("Processing dataset with sliding windows (this may take a moment)...")
    processed_dataset = raw_dataset.map(
        lambda x: sliding_window_preprocess(x, tokenizer),
        batched=True,
        batch_size=100,
        remove_columns=raw_dataset.column_names,  # Remove old columns, keep only inputs/labels
        num_proc=4
    )

    print(f"Total processed chunks: {len(processed_dataset)}")

    # 4. Create Splits (Smaller Val/Test as requested)
    # Split train and (temp_test)
    split_1 = processed_dataset.train_test_split(test_size=2000, seed=42)
    train_dataset = split_1["train"]
    temp_test = split_1["test"]

    # Split temp_test into val and test (1000 each)
    split_2 = temp_test.train_test_split(test_size=0.5, seed=42)
    eval_dataset = split_2["train"]
    test_dataset = split_2["test"]

    print(f"Train size: {len(train_dataset)}, Val size: {len(eval_dataset)}, Test size: {len(test_dataset)}")

    # 5. Training Arguments with TensorBoard logging
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        eval_steps=500,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=1000,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=EPOCHS,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",  # Log to TensorBoard
        logging_dir=LOGGING_DIR,  # Directory for logs
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        tokenizer=tokenizer,
    )

    # 6. Train
    print("Starting training...")
    trainer.train()

    # 7. Save
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save the test set to disk for the evaluation script to use
    print("Saving test dataset for evaluation...")
    test_dataset.save_to_disk(f"{DATASET_DIR}/test_data")


if __name__ == "__main__":
    main()