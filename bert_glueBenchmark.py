import os
import argparse
import numpy as np
from datasets import load_dataset, DatasetDict
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))



def fine_tune_and_eval(model_name, task, epochs, batch):

    print(f"\n===== Training {model_name} on {task} =====")

    # Since test set labels unavailable, use train data
    # 0.8 as train, 0.1 as validation, 0.1 as test 
    if task == "mnli_matched":
        full_train = load_dataset("glue", "mnli", split="train")
        full_test = load_dataset("glue", "mnli", split="validation_matched")
    elif task == "mnli_mismatched":
        full_train = load_dataset("glue", "mnli", split="train")
        full_test = load_dataset("glue", "mnli", split="validation_mismatched")
    else:
        full_train = load_dataset("glue", task, split="train")
        full_test = load_dataset("glue", task, split="validation")
    #l = len(full_train)
    #a = int(l * 0.8)
    #b = int(l * 0.9)

    #train_80 = full_train.select(range(0, a))
    #dev_10   = full_train.select(range(a, b))
    #test_10  = full_train.select(range(b, l))
    raw_dataset = DatasetDict({
        "train": full_train,
        # "validation": full_test,
        "test": full_test
    })

    # raw_dataset = load_dataset("glue", task)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Sentence keys are either one or two sentences
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

    def preprocess(batch):
        if sentence2_key is None:
            return tokenizer(batch[sentence1_key], truncation=True, padding="max_length")
        return tokenizer(
            batch[sentence1_key],
            batch[sentence2_key],
            truncation=True,
            padding="max_length",
        )

    tokenized = raw_dataset.map(preprocess, batched=True)

    num_labels = (
        1 if task == "stsb" else raw_dataset["train"].features["label"].num_classes
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir=f"./checkpoints/{model_name.replace('/', '_')}_{task}",
        learning_rate=2e-5,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,

        do_train=True,
        do_eval=False,
    )

    # GLUE metrics
    metric = evaluate.load("glue", "mnli" if "mnli" in task else task)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if task == "stsb":
            preds = np.squeeze(logits)
        else:
            preds = np.argmax(logits, axis=1)
        return metric.compute(predictions=preds, references=labels)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        #eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # print(f"\n===== Validation Results ({model_name}) =====")
    # results = trainer.evaluate()
    # print(results)

    print(f"\n===== Test Set Results ({model_name}) =====")
    test_results = trainer.predict(tokenized["test"])

    if task == "stsb":
        preds = np.squeeze(test_results.predictions)
    else:
        preds = np.argmax(test_results.predictions, axis=1)

    test_metrics = metric.compute(predictions=preds, references=test_results.label_ids)
    print(test_metrics)
    print("test metrics done")

    # save model as name_task_new e.g. bert-base-uncased_mrpc
    save_name = f"{model_name.replace('/', '_')}_{task}_new"
    save_dir = os.path.join("checkpoints", save_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nSaving model to {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    return test_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        help="GLUE task: sst2, mrpc, qqp, mnli_matched, mnli_mismatched, rte, cola, stsb, wnli, qnli")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32)

    args = parser.parse_args()

    # list of models to compare
    model_list = [
        "bert-base-uncased",
        "felflare/bert-restore-punctuation",
        "xlm-roberta-base",
        "oliverguhr/fullstop-punctuation-multilingual-base"
    ]

    all_results = {}

    for model in model_list:
        results = fine_tune_and_eval(model, args.task, args.epochs, args.batch)
        all_results[model] = results

    print("\n======== FINAL COMPARISON ========")
    for model, res in all_results.items():
        print(f"\n{model}:\n{res}")
