import json
import logging
import torch

from dataclasses import TrainData
from datasets import Dataset
from lightning import LightningModule
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    PreTrainedModel, T5ForConditionalGeneration,
    T5TokenizerFast,
    get_scheduler,
)
from typing import Callable

logging.basicConfig(
    filename = 'logs/t5',
    filemode = 'w',
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level = logging.DEBUG,
)
logger = logging.getLogger(__name__)


class PRT5(LightningModule):
    """PR-T5 model"""
    
    def __init__(
        self,
        adam_epsilon: float,
        eval_batch_size: int,
        learning_rate: float,
        max_seq_length: int,
        num_train_epochs: int,
        num_workers: int,
        train_batch_size: int,
        training_data: TrainData,
        warmup_steps: int,
        weight_decay: float,
        epoch_end_result_path: str = 'test_predictions.jsonl',
        model: str = 'google-t5/t5-base',
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        self.tokenizer = T5TokenizerFast.from_pretrained(model)
        self.outputs = []
    
    def __getitem__(self, item: str) -> any:
        return getattr(self.hparams, item)
    
    def configure_optimizers(self) -> dict[str, object]:
        optimizer = AdamW(
            self.parameters(),
            lr = self['learning_rate'],
            eps = self['adam_epsilon'],
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': get_scheduler(
                    name = 'linear',
                    optimizer = optimizer,
                    num_warmup_steps = self['warmup_steps'],
                    num_training_steps = self.trainer.estimated_stepping_batches,
                ),
                'interval': 'step',
                'frequency': 1
            },
        }
    
    def forward(self, input_ids, attention_mask, labels = None) -> PreTrainedModel:
        return self.model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
    
    def _generic_step(self, batch, logged_name = 'loss') -> torch.Tensor:
        """Template step"""
        labels = batch['labels']
        labels[labels == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids = batch['input_ids'],
            attention_mask = batch['attention_mask'],
            labels = labels
        )
        loss = outputs.loss
        self.log(logged_name, loss, prog_bar = True, sync_dist = True)
        return loss
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._generic_step(batch, logged_name = 'train_loss')
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self._generic_step(batch, logged_name = 'val_loss')
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # Generate predictions
        generated_ids = self.model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            max_length = labels.shape[1],
            num_beams = 4  # or whatever decoding strategy you prefer
        )
        
        # Decode to text
        predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens = True)
        targets = self.tokenizer.batch_decode(labels, skip_special_tokens = True)
        
        self.outputs += [{'predictions': predictions, 'targets': targets}]
    
    def on_test_epoch_end(self):
        logger.debug('test epoch ended')
        all_predictions = []
        all_targets = []
        for output in self.outputs:
            all_predictions.extend(output['predictions'])
            all_targets.extend(output['targets'])
        all_sources = self.train_dataloader().dataset['source']
        with open(self['epoch_end_result_path'], 'w') as f:
            for prediction, target, source in zip(all_predictions, all_targets, all_sources):
                f.write(
                    json.dumps(
                        {
                            'prediction': prediction,
                            'target': target,
                            'source': source,
                        }
                    ) + '\n'
                )
    
    def _generic_dataloader(self, split: str) -> DataLoader:
        return self['training_data'].loader(
            split,
            tokenizer = self.tokenizer,
            eval_batch_size = self['eval_batch_size'],
            max_seq_length = self['max_seq_length'],
            num_workers = self['num_workers']
        )
    
    def train_dataloader(self) -> DataLoader:
        return self._generic_dataloader(split = 'train')
    
    def val_dataloader(self) -> DataLoader:
        return self._generic_dataloader(split = 'dev')
    
    def test_dataloader(self) -> DataLoader:
        return self._generic_dataloader(split = 'test')
    
    def decoder(
        self,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> Callable[[str], str]:
        def decode(ids):
            return self.tokenizer.decode(
                ids,
                skip_special_tokens = skip_special_tokens,
                clean_up_tokenization_spaces = clean_up_tokenization_spaces
            ).strip()
        
        return decode
    
    def generate(
        self,
        batch_size: int,
        input_dataset: Dataset,
        max_len: int = 256,
        num_beams: int = 4,
        num_workers: int = 8,
        shuffle: bool = True,
        skip_special_tokens: bool = True,
    ) -> tuple[list[str], list[str], list[str]]:
        self.model.eval()
        model = self.to('cuda')
        outputs = []
        targets = []
        texts = []
        for batch in tqdm(
            DataLoader(
                input_dataset,
                batch_size = batch_size,
                num_workers = num_workers,
                shuffle = shuffle,
            )
        ):
            outputs.extend(
                map(
                    self.decoder(skip_special_tokens),
                    model.model.generate(
                        input_ids = batch['input_ids'].to('cuda'),
                        attention_mask = batch['attention_mask'].to('cuda'),
                        max_length = max_len,
                        num_beams = num_beams,
                    )
                )
            )
            targets.extend(map(self.decoder(skip_special_tokens), batch['labels']))
            texts.extend(map(self.decoder(skip_special_tokens), batch['input_ids']))
        return texts, outputs, targets
    
    def save(self, path: str):
        """Save model parameters to path"""
        torch.save(self.state_dict(), path)
        logger.info(f'Saved model to {path}')