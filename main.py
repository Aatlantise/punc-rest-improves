import argparse
import logging
import os
import random
import json
import numpy as np
from tqdm import tqdm
from typing import List, Any, Union
from datasets import Dataset
from eval import pr_score, NER_eval

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import torch
from torch import autograd, nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers.models.t5.modeling_t5 import T5Config
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    get_scheduler
)

autograd.set_detect_anomaly(True)
logger = logging.getLogger(__name__)


def flatten(nested_list: List[List[Any]]) -> List[Any]:
    """
    Flattens nested list

    :param nested_list: nested list of rank 2
    :return: flattened list
    """
    flat_list = []
    for sublist in nested_list:
        flat_list.extend(sublist)
    return flat_list


def set_seed(seed):
    """
    Sets seed to given number, used to control seed in multi-seed evaluation

    :param seed:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_pr_dataset(dataset_name:str):
    # name should be a jsonl file. Originall "punctuation_restoration_dataset.jsonl"
    with open(dataset_name) as f:
        data = []
        for line in f:
            data.append(json.loads(line))

    a = int(len(data) * 0.8)
    b = int(len(data) * 0.9)
    train = data[:a]
    dev = data[a:b]
    test = data[b:]

    return {"train": train,
            "dev": dev,
            "test": test
            }

class T5ClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks. Source code from
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
    """

    def __init__(self, config: T5Config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class PRT5(pl.LightningModule):
    def __init__(self, model_name_or_path: str,
                 learning_rate: float,
                 weight_decay: float,
                 adam_epsilon: float,
                 warmup_steps: int,
                 max_seq_length: int,
                 train_batch_size: int,
                 eval_batch_size: int,
                 num_train_epochs: int,
                 dataset_name: str,
                 ):
        super().__init__()
        self.save_hyperparameters()
        
        print(f"\n The model initialised is (model name or path): {model_name_or_path} \n")

        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name_or_path)
        self.test_dl = DataLoader(
            get_punctuation_dataset(self.tokenizer, "test", self.hparams.max_seq_length, dataset_name),
            batch_size=self.hparams.eval_batch_size,
            num_workers=4,
        )
        self.outputs = []

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def on_test_epoch_end(self):
        all_preds = []
        all_targets = []
        all_sources = self.test_dl.dataset["source"]

        for output in self.outputs:
            all_preds.extend(output["preds"])
            all_targets.extend(output["targets"])

        # Now evaluate using your custom metric
        score = pr_score(all_sources, all_preds, all_targets)
        self.log("PR F1 Score: ", score)

        # Optionally, write to file for inspection
        #  with open("test_predictions.jsonl", "w") as f:
        #      for pred, target in zip(all_preds, all_targets):
        #          f.write(json.dumps({"prediction": pred, "target": target}) + "\n")

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Generate predictions
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=labels.shape[1],
            num_beams=4  # or whatever decoding strategy you prefer
        )

        # Decode to text
        preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        targets = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        self.outputs += [{"preds": preds, "targets": targets}]

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels
        )
        val_loss = outputs.loss
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        num_training_steps = self.trainer.estimated_stepping_batches
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1
            },
        }

    def train_dataloader(self):
        return DataLoader(
            get_punctuation_dataset(self.tokenizer, "train", self.hparams.max_seq_length, self.hparams.dataset_name),
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            get_punctuation_dataset(self.tokenizer, "dev", self.hparams.max_seq_length, self.hparams.dataset_name),
            batch_size=self.hparams.eval_batch_size,
            num_workers=4,
        )

    def test_dataloader(self):
        return self.test_dl

def get_punctuation_dataset(tokenizer, split: str, max_len: int, dataset_name: str):
    # Load your dataset file, e.g., .pkl or .jsonl
    data = load_pr_dataset(dataset_name)[split]  # or custom loading logic

    def preprocess(example):
        inputs = tokenizer(example['source'], max_length=max_len, truncation=True, padding="max_length")
        targets = tokenizer(example['target'], max_length=max_len, truncation=True, padding="max_length")
        inputs['labels'] = targets['input_ids']
        return inputs

    dataset = Dataset.from_list(data).map(preprocess, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset

def get_punctuation_dataset_hardcode(tokenizer, split: str, max_len: int):
    return get_punctuation_dataset(tokenizer, split, max_len, "punctuation_restoration_dataset.jsonl")

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))


def run(
    model_name_or_path: str = "t5-base",
    output_dir: str = "./outputs",
    max_epochs: int = 3,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    adam_epsilon: float = 1e-8,
    warmup_steps: int = 0,
    max_seq_length: int = 256,
    seed: int = 42,
    precision: str = "bf16",  # or "32-true" for FP32
    num_workers: int = 4,
    accelerator: str = "gpu",
    devices: int = 1,
    save_top_k: int = 1,
    monitor_metric: str = "val_loss",
    log_every_n_steps: int = 10,
    resume_from_checkpoint: str = None,
    NER_eval_flag: bool = False,
    dataset_name: str = "processed_conll03.jsonl",
    evaluate_only: bool = False,
):
    pl.seed_everything(seed)
    set_seed(seed)
    print("running on model", model_name_or_path)

    if evaluate_only:
        print("only running NER evaluation:")
        run_NER(dataset_name, output_dir=output_dir, eval_batch_size=32, max_seq_length=256)
        return

    # Model
    model = PRT5(
        model_name_or_path=model_name_or_path,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        adam_epsilon=adam_epsilon,
        warmup_steps=warmup_steps,
        max_seq_length=max_seq_length,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_train_epochs=max_epochs,
        dataset_name=dataset_name,
    )

    # Logging
    logger = TensorBoardLogger(save_dir=output_dir, name="logs")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="{epoch}-{val_loss:.4f}",
        save_top_k=save_top_k,
        verbose=True,
        monitor=monitor_metric,
        mode="min",
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=precision,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=log_every_n_steps,
        default_root_dir=output_dir,
    )

    # Training
    trainer.fit(model)

    # Optionally test on test set (implement test_dataloader in the module first)
    if hasattr(model, 'test_dataloader'):
        trainer.test(model)
    
    # Get NER evaluation
    if NER_eval_flag:
        print("NER evaluation running: ")
        run_NER(dataset_name, output_dir, eval_batch_size=32, max_seq_length=256)

def run_NER(dataset_name: str, output_dir: str, eval_batch_size: int=32, max_seq_length: int =256):
    print("NER evaluation \n")
    ckpt_path = os.path.join(output_dir, "checkpoints", "last.ckpt")
    model = PRT5.load_from_checkpoint(ckpt_path)
    texts, outputs, targets = generate(
        ckpt = None, # change to checkpoint if desired
        # ckpt = ckpt_path,
        model=model,
        input_dataset=get_punctuation_dataset(model.tokenizer, "test", max_seq_length, dataset_name),
        tokenizer=model.tokenizer,
        batch_size=eval_batch_size,
        # max_len=max_seq_length,
        shuffle=False
    )
    #  NER_eval(texts, outputs, targets, printer=logger.info)
    NER_eval(texts, outputs, targets)

def generate(ckpt: Union[str, None], model, input_dataset, tokenizer, batch_size, max_len=256, num_beams=4, skip_special_tokens=True, shuffle=True):
    if ckpt is not None:
        model = model.load_from_checkpoint(ckpt)
    dataloader = DataLoader(input_dataset, batch_size=batch_size, num_workers=8, shuffle=shuffle)
    model.model.eval()
    model = model.to("cuda")

    # loop through dataloader
    outputs = []
    targets = []
    texts = []
    for batch in tqdm(dataloader):
        outs = model.model.generate(input_ids=batch['input_ids'].to("cuda"),
                                    attention_mask=batch['attention_mask'].to("cuda"),
                                    max_length=max_len, num_beams=num_beams
                                    )
        dec = [tokenizer.decode(ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False).strip() for ids in
               outs]
        target = [tokenizer.decode(ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False).strip()
                  for ids in batch["labels"]]
        text = [tokenizer.decode(ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False).strip()
                for ids in batch["input_ids"]]
        texts.extend(text)
        outputs.extend(dec)
        targets.extend(target)

    return texts, outputs, targets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="t5-base")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--NER_eval_flag", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="processed_conll03.jsonl")
    parser.add_argument("--evaluate_only", action="store_true")
    args = parser.parse_args()
    print("inputs are: ", vars(args))
    run(**vars(args))
