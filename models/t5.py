import torch

from data.modules import TrainData
from lightning import LightningModule
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    get_scheduler,
)
from typing import Callable, Union
from utils import logger, progress

logger = logger()


class PRT5(LightningModule):
    """PR-T5 model"""
    
    def __init__(
        self,
        adam_epsilon: float,
        eval_batch_size: int,
        learning_rate: float,
        max_seq_length: int,
        num_train_epochs: int,
        train_batch_size: int,
        warmup_steps: int,
        weight_decay: float,
        epoch_end_result_path: str = 'test_predictions.jsonl',
        model: str = 'google-t5/t5-base',
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        self.tokenizer = T5TokenizerFast.from_pretrained(model)
        self.outputs = []
        self.training_data: Union[TrainData, None] = None
    
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
    
    def on_test_epoch_end(self):
        logger.info('Test epoch ended')
    
    def store_data(self, d: TrainData):
        """Store a TrainData with a model instance for dataloader methods"""
        self.training_data = d
    
    def _verify_data_stored(self):
        if self.training_data is None:
            raise Exception('PRT5 model has no stored TrainData data. Call .store_data() before using dataloaders!')
        
    def _generic_dataloader(self, split: str) -> DataLoader:
        self._verify_data_stored()
        return self.training_data.loader(
            split,
            tokenizer = self.tokenizer,
            eval_batch_size = self['eval_batch_size'],
            max_seq_length = self['max_seq_length'],
            num_workers = self['num_workers'],
        )
        
    def train_dataloader(self) -> DataLoader:
        return self._generic_dataloader(split = 'train')
        
    def test_dataloader(self) -> DataLoader:
        return self._generic_dataloader(split = 'test')
    
    def val_dataloader(self) -> DataLoader:
        return self._generic_dataloader(split = 'dev')
    
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
        
    def save(self, path: str):
        """Save model parameters to path"""
        torch.save(self.state_dict(), path)
        logger.info(f'Saved model to {path}')
    
    def generate(
        self,
        input_dataloader: DataLoader,
        max_len: int = 512,
        num_beams: int = 4,
        skip_special_tokens: bool = True,
    ) -> tuple[list[str], list[str], list[str]]:
        self.model.eval()
        cuda_plm = self.to('cuda')
        texts, outputs, targets = [], [], []
        with torch.no_grad():
            for batch in progress(input_dataloader, 'Generating texts'):
                texts.extend(map(
                    cuda_plm.decoder(skip_special_tokens),
                    batch['input_ids']
                ))
                outputs.extend(map(
                    cuda_plm.decoder(skip_special_tokens),
                    cuda_plm.model.generate(
                        input_ids = batch['input_ids'].to('cuda'),
                        attention_mask = batch['attention_mask'].to('cuda'),
                        max_length = max_len,
                        num_beams = num_beams,
                    )
                ))
                targets.extend(map(
                    cuda_plm.decoder(skip_special_tokens),
                    batch['labels']
                ))
        return texts, outputs, targets
    
    def test(self, eval_metric: Callable[[list[str], list[str], list[str]], tuple[float, float, float]]):
        if not eval_metric:
            logger.info(f'No metric provided for in-fitting evaluations, continuing. ')
            return
            
        logger.info('Testing model during fitting. ')
        
        logger.info('Loading test dataloader')
        test_loader = self.test_dataloader()
        
        logger.info('Generating outputs')
        texts, outputs, targets = self.generate(test_loader)
        
        logger.info('Evaluating current epoch')
        p, r, f1 = eval_metric(texts, outputs, targets)
        
        logger.info('Evaluation complete. Results:')
        logger.info(f'\tPrecision: \t{p:.4f}')
        logger.info(f'\tRecall: \t{r:.4f}')
        logger.info(f'\tF1: \t{f1:.4f}\n')
