import logging
import os
import random
import json
import numpy as np
# from tqdm import tqdm
# from typing import List, Any, Union
from datasets import Dataset

# Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Torch
import torch
from torch import autograd, nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Transformers
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    get_scheduler
)

autograd.set_detect_anomaly(True)
logger = logging.getLogger(__name__)

# set seed for all random components
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# tokenize dataset for training
def tokenize(data, tokenizer, split: str, max_len: int):
    def preprocess(example):
        sources = tokenizer(
            example['source'],
            max_length=max_len,
            truncation=True,
            padding='max_length',
        )
        targets = sources = tokenizer(
            example['target'],
            max_length=max_len,
            truncation=True,
            padding='max_length',
        )
        sources['labels'] = targets['input_ids']
        return sources
    ds = Dataset.from_list(data).map(preprocess, batched=True)
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return ds

# split dataset for training, validating, and testing
class DataForTraining:
    def __init__(self, jsonl_path: str):
        with open(jsonl_path) as jsonl_file:
            data = []
            for line in jsonl_file:
                data.append(json.loads(line))
        l = len(data)
        a = int(l * 0.8)
        b = int(l * 0.9)
        self.data = {
            'train': data[:a],
            'dev': data[a:b],
            'test': data[b:]
        }

    def loader(self, split: str, tokenizer, max_seq_length, eval_batch_size, num_workers):
        def preprocess(example):
            sources = tokenizer(
                example['source'],
                max_length=max_seq_length,
                truncation=True,
                padding='max_length',
            )
            targets = tokenizer(
                example['target'],
                max_length=max_seq_length,
                truncation=True,
                padding='max_length',
            )
            sources['labels'] = targets['input_ids']
            return sources
        ds = Dataset.from_list(self.data[split]).map(preprocess, batched=True)
        ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        return DataLoader(ds, batch_size=eval_batch_size, num_workers=num_workers)

# PR-T5 model
class PRT5(pl.LightningModule):
    def __init__(self,
        adam_epsilon: float,
        eval_batch_size: int,
        learning_rate: float,
        max_seq_length: int,
        model: str,
        num_train_epochs: int,
        num_workers: int,
        train_batch_size: int,
        training_data: DataForTraining,
        warmup_steps: int,
        weight_decay: float,
        epoch_end_result_path: str = 'test_predictions.jsonl',
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        self.tokenizer = T5TokenizerFast.from_pretrained(model)
        self.outputs = []
        self.epoch_end_result_path = epoch_end_result_path

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            eps=self.hparams.adam_epsilon
        )
        num_training_steps = self.trainer.estimated_stepping_batches
        lr_scheduler = get_scheduler(
            name='linear',
            optimizer=optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=num_training_steps,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1
            },
        }

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        labels = batch['labels']
        labels[labels == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=labels
        )
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        labels = batch['labels']
        labels[labels == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=labels
        )
        val_loss = outputs.loss
        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        # Generate predictions
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=labels.shape[1],
            num_beams=4  # or whatever decoding strategy you prefer
        )

        # Decode to text
        predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        targets = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        self.outputs += [{'predictions': predictions, 'targets': targets}]

    def on_test_epoch_end(self):
        all_predictions = []
        all_targets = []
        for output in self.outputs:
            all_predictions.extend(output['predictions'])
            all_targets.extend(output['targets'])
        all_sources = self.test_dl.dataset['source']
        with open(self.epoch_end_result_path, 'w') as f:
            for prediction, target, source in zip(all_predictions, all_targets, all_sources):
                f.write(json.dumps({
                    'prediction': prediction,
                    'target': target,
                    'source': source,
                }) + '\n')

    def _generic_dataloader(self, split: str):
        if not self.hparams.training_data:
            raise Exception('Data not loaded!')
        return self.hparams.training_data.loader(
            split,
            tokenizer=self.tokenizer,
            eval_batch_size=self.hparams.eval_batch_size,
            max_seq_length=self.hparams.max_seq_length,
            num_workers=self.hparams.num_workers
        )

    def train_dataloader(self):
        self._generic_dataloader(split='train')

    def val_dataloader(self):
        self._generic_dataloader(split='dev')

    def test_dataloader(self):
        self._generic_dataloader(split='test')

# train
def run(
    accelerator: str = 'gpu',
    adam_epsilon: float = 1e-8,
    data_path: str = 'punctuation_restoration_data.jsonl',
    devices: int = 1,
    eval_batch_size: int = 32,
    learning_rate: float = 3e-4,
    log_every_n_steps: int = 10,
    max_epochs = 3,
    max_seq_length: int = 256,
    model = 't5_base',
    monitor_metric: str = 'val_loss',
    num_train_epochs: int = 3,
    num_workers: int = 4,
    output_dir = './outputs',
    precision: str = 'bf16',
    # resume_from_checkpoint: str = None,
    save_top_k: int = 1,
    seed: int = 42,
    train_batch_size: int = 32,
    warmup_steps: int = 0,
    weight_decay: float = 0.01,
):
    set_seed(seed)
    training_data = DataForTraining(data_path)
    model = PRT5(
        adam_epsilon=adam_epsilon,
        eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        model=model,
        num_train_epochs=num_train_epochs,
        num_workers=num_workers,
        train_batch_size=train_batch_size,
        training_data=training_data,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=TensorBoardLogger(save_dir=output_dir, name='logs'),
        callbacks=[
            ModelCheckpoint(
                dirpath=os.path.join(output_dir, 'checkpoints'),
                filename='{epoch}-{val_loss:.4f}',
                save_top_k=save_top_k,
                verbose=True,
                monitor=monitor_metric,
                mode='min',
            ),
            LearningRateMonitor(logging_interval='step')
        ],
        precision=precision,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=log_every_n_steps,
        default_root_dir=output_dir,
    )
    trainer.fit(model)
    trainer.test(model)

if __name__ == '__main__':
    run(data_path='datasets/conll-2012-srl.jsonl')
