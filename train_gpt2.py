import re
import torch
import numpy as np
from collections import defaultdict
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

# --- Configuration ---
MODEL_CHECKPOINT = "gpt2-medium"
DATASET_NAME = "data/finance-news"
MAX_SEQ_LENGTH = 1024
TARGET_CHUNK_SIZE = 384
MAX_NEW_TOKENS = 448
OVERLAP = 100
STRIDE = TARGET_CHUNK_SIZE - OVERLAP
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 32
LEARNING_RATE = 5e-5
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 3
OUTPUT_DIR = "./pr-gpt2-output"
LOGGING_DIR = "./logs-gpt2"

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


def make_feature(chunk_ids, tokenizer):
    """
    Convert one target chunk into a causal-LM prompt/completion example.

    GPT-2 receives one combined sequence:
        restore punctuation: <corrupted text> restored: <target text><eos>

    Prompt labels are -100, so loss is computed only on the restored text.
    """
    while len(chunk_ids) >= 20:
        target_text = clean_text(
            tokenizer.decode(chunk_ids, skip_special_tokens=True)
        )

        source_text = (
            "restore punctuation: "
            + corruption_function(target_text)
            + " restored: "
        )

        prompt_ids = tokenizer(
            source_text,
            add_special_tokens=False,
        )["input_ids"]

        target_ids = tokenizer(
            target_text,
            add_special_tokens=False,
        )["input_ids"] + [tokenizer.eos_token_id]

        combined_length = len(prompt_ids) + len(target_ids)

        if (
            combined_length <= MAX_SEQ_LENGTH
            and len(target_ids) <= MAX_NEW_TOKENS
            and len(prompt_ids) + MAX_NEW_TOKENS <= MAX_SEQ_LENGTH
        ):
            return {
                "input_ids": prompt_ids + target_ids,
                "labels": [-100] * len(prompt_ids) + target_ids,
            }

        # Shrink only when the prompt/completion does not fit GPT-2's context.
        chunk_ids = chunk_ids[:-16]

    return None


def sliding_window_preprocess(examples, tokenizer):
    model_inputs = {"input_ids": [], "labels": []}

    for text in examples["text"]:
        cleaned_text = clean_text(text)
        if len(cleaned_text) < 10:
            continue

        full_token_ids = tokenizer(
            cleaned_text,
            add_special_tokens=False,
        )["input_ids"]

        for i in range(0, len(full_token_ids), STRIDE):
            chunk_ids = full_token_ids[i:i + TARGET_CHUNK_SIZE]

            if len(chunk_ids) < 20:
                continue

            feature = make_feature(chunk_ids, tokenizer)
            if feature is not None:
                model_inputs["input_ids"].append(feature["input_ids"])
                model_inputs["labels"].append(feature["labels"])

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
                if t_punct != "":
                    stats[t_punct]["FN"] += 1
                if p_punct != "":
                    stats[p_punct]["FP"] += 1

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

    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(
        preds != -100,
        preds,
        tokenizer.pad_token_id,
    )
    labels = np.where(
        labels != -100,
        labels,
        tokenizer.pad_token_id,
    )

    decoded_preds = tokenizer.batch_decode(
        preds,
        skip_special_tokens=True,
    )
    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True,
    )

    for i in range(min(3, len(decoded_preds))):
        print(f"Predicted: {decoded_preds[i]}")
        print(f"Target: {decoded_labels[i]}")

    # Aggregators
    total_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    total_aligned_words = 0

    # Iterate over batch
    for pred_text, target_text in zip(decoded_preds, decoded_labels):
        batch_stats, batch_words = get_stats_for_pair(
            target_text,
            pred_text,
        )
        total_aligned_words += batch_words

        for punct, counts in batch_stats.items():
            total_stats[punct]["TP"] += counts["TP"]
            total_stats[punct]["FP"] += counts["FP"]
            total_stats[punct]["FN"] += counts["FN"]

    # Calculate scores
    results = {}
    macro_f1_scores = []

    for punct in sorted(TARGET_PUNCT_SET):
        tp = total_stats[punct]["TP"]
        fp = total_stats[punct]["FP"]
        fn = total_stats[punct]["FN"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results[f"f1_{punct}"] = f1
        macro_f1_scores.append(f1)

    results["macro_f1"] = (
        sum(macro_f1_scores) / len(macro_f1_scores)
        if macro_f1_scores
        else 0.0
    )

    return results


class GPT2GenerationTrainer(Trainer):
    """
    Trainer that mirrors Seq2SeqTrainer's generated-prediction behavior.

    Standard Trainer returns causal-LM logits. For punctuation metrics, this
    class instead generates from the masked prompt and returns generated token
    IDs alongside the target token IDs.
    """

    @staticmethod
    def _pad(sequences, pad_token_id, device):
        max_length = max(len(sequence) for sequence in sequences)

        result = torch.full(
            (len(sequences), max_length),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )

        for i, sequence in enumerate(sequences):
            result[i, :len(sequence)] = sequence

        return result

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
            loss = outputs.loss.mean().detach()

        if prediction_loss_only:
            return loss, None, None

        prompts = []
        targets = []

        for input_ids, attention_mask, labels in zip(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["labels"],
        ):
            valid_length = int(attention_mask.sum().item())
            input_ids = input_ids[:valid_length]
            labels = labels[:valid_length]

            target_positions = (labels != -100).nonzero(
                as_tuple=False
            ).flatten()

            if target_positions.numel() == 0:
                raise ValueError(
                    "Evaluation example has no supervised target tokens."
                )

            target_start = target_positions[0].item()
            prompts.append(input_ids[:target_start])
            targets.append(labels[target_start:])

        # Decoder-only batched generation requires left-padded prompts.
        max_prompt_length = max(len(prompt) for prompt in prompts)

        prompt_ids = torch.full(
            (len(prompts), max_prompt_length),
            self.processing_class.pad_token_id,
            dtype=torch.long,
            device=inputs["input_ids"].device,
        )
        prompt_mask = torch.zeros_like(prompt_ids)

        for i, prompt in enumerate(prompts):
            prompt_ids[i, -len(prompt):] = prompt
            prompt_mask[i, -len(prompt):] = 1

        generation_model = self.accelerator.unwrap_model(model)

        with torch.no_grad():
            generated = generation_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.processing_class.pad_token_id,
                eos_token_id=self.processing_class.eos_token_id,
                use_cache=True,
            )

        # Decoder-only generation returns prompt + generated continuation.
        generated = generated[:, max_prompt_length:]

        generated = self._pad(
            list(generated),
            self.processing_class.pad_token_id,
            generated.device,
        )
        targets = self._pad(
            targets,
            self.processing_class.pad_token_id,
            generated.device,
        )

        return loss, generated, targets


def main(debug=False):
    # 1. Setup
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CHECKPOINT,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CHECKPOINT
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # 2. Load and Process Dataset
    print("Loading and processing data...")

    epochs = EPOCHS

    if debug:
        dataset = load_from_disk(DATASET_NAME).select(range(1000))
        epochs = 3
    else:
        dataset = load_from_disk(DATASET_NAME)

    processed_dataset = dataset.map(
        lambda x: sliding_window_preprocess(x, tokenizer),
        batched=True,
        batch_size=100,
        remove_columns=dataset.column_names,
        num_proc=4,
    )

    # Split: Train vs Eval
    split = processed_dataset.train_test_split(
        test_size=0.01,
        seed=42,
    )
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # evaluation takes too long, 18 hours and didn't finish
    MAX_EVAL_FEATURES = 1000

    if len(eval_dataset) > MAX_EVAL_FEATURES:
        eval_dataset = (
            eval_dataset
            .shuffle(seed=42)
            .select(range(MAX_EVAL_FEATURES))
        )
    print(f"Training features: {len(train_dataset)}")
    print(f"Evaluation features: {len(eval_dataset)}")

    bf16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    # 3. Training Args
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",       # Evaluate every epoch
        save_strategy="epoch",       # Save every epoch
        logging_strategy="steps",
        logging_steps=500,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        weight_decay=0.01,
        num_train_epochs=epochs,
        fp16=torch.cuda.is_available() and not bf16,
        bf16=bf16,
        report_to="tensorboard",
        logging_dir=LOGGING_DIR,

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        # Avoid retaining every epoch checkpoint
        save_total_limit=2,
    )

    patience = EARLY_STOPPING_PATIENCE
    # 4. Initialize Trainer with compute_metrics
    trainer = GPT2GenerationTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
        ),
        processing_class=tokenizer,
        compute_metrics=lambda eval_preds: compute_metrics(
            eval_preds,
            tokenizer,
        ),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=patience,
                early_stopping_threshold=0.0,
            )
        ],
    )

    # 5. Train
    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.model.config.use_cache = True
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main(debug=False)