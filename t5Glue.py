import os
import argparse
import random

import evaluate
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)


print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

MAX_LEN = 128


def fine_tune_and_eval(model_name, task, epochs, batch):
    print(f"\n===== Training {model_name} on {task} =====")

    if task == "mnli_matched":
        full_train = load_dataset("glue", "mnli", split="train")
        full_test = load_dataset("glue", "mnli", split="validation_matched")
    elif task == "mnli_mismatched":
        full_train = load_dataset("glue", "mnli", split="train")
        full_test = load_dataset("glue", "mnli", split="validation_mismatched")
    else:
        full_train = load_dataset("glue", task, split="train")
        full_test = load_dataset("glue", task, split="validation")

    raw_dataset = DatasetDict({
        "train": full_train,
        "test": full_test,
    })

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    task_to_keys = {
        "sst2": ("sentence", None),
        "mrpc": ("sentence1", "sentence2"),
        "qqp": ("question1", "question2"),
        "mnli_matched": ("premise", "hypothesis"),
        "mnli_mismatched": ("premise", "hypothesis"),
        "qnli": ("question", "sentence"),
        "rte": ("sentence1", "sentence2"),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
        "cola": ("sentence", None),
    }

    sentence1_key, sentence2_key = task_to_keys[task]

    def preprocess(examples):
        if sentence2_key is None:
            return tokenizer(
                examples[sentence1_key],
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
            )
        return tokenizer(
            examples[sentence1_key],
            examples[sentence2_key],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )

    tokenized = raw_dataset.map(preprocess, batched=True)

    num_labels = (
        1 if task == "stsb"
        else raw_dataset["train"].features["label"].num_classes
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=f"./checkpoints/{model_name.replace('/', '_')}_{task}",
        learning_rate=1e-5,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="no",
        save_strategy="no",
        do_train=True,
        do_eval=False,
        seed=SEED,
        data_seed=SEED,
    )

    metric = evaluate.load("glue", "mnli" if "mnli" in task else task)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
    )

    trainer.train()

    print(f"\n===== Test Set Results ({model_name}) =====")
    test_results = trainer.predict(tokenized["test"],ignore_keys=["past_key_values", "encoder_last_hidden_state"])

    logits = test_results.predictions

    if isinstance(logits, tuple):
        logits = logits[0]
    
    if task == "stsb":
        predictions = np.squeeze(logits)
    else:
        predictions = np.argmax(logits, axis=1)

    test_metrics = metric.compute(
        predictions=predictions,
        references=test_results.label_ids,
    )
    print(test_metrics)
    print("test metrics done")

    save_name = f"{model_name.replace('/', '_')}_{task}_final"
    save_dir = os.path.join("checkpoints", save_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nSaving model to {save_dir} : NOT SAVING - UNCOMMENT BELOW TO SAVE")
    # model.save_pretrained(save_dir)
    # tokenizer.save_pretrained(save_dir)

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help=(
            "GLUE task: sst2, mrpc, qqp, mnli_matched, mnli_mismatched, "
            "rte, cola, stsb, wnli, qnli"
        ),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    # Use "checkpoints/T5_accumulation" for the PR-T5 checkpoint.
    model_list = [
        # "T5_accumulation"
        "google-t5/t5-base"
    ]

    all_results = {}

    for model_name in model_list:
        all_results[model_name] = {}

        for seed in [1, 2, 3]:
            SEED = seed
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            set_seed(SEED)

            print(f"seed {SEED}")
            results = fine_tune_and_eval(
                model_name,
                args.task,
                args.epochs,
                args.batch,
            )
            all_results[model_name][seed] = results

    print("\n======== FINAL RESULTS ========")

    for model_name, seed_results in all_results.items():
        for seed, results in seed_results.items():
            print(f"{model_name}, seed {seed}: {results}")