import torch
import torch.nn as nn

from data.modules import TrainData, IntTrainData
from lightning import LightningModule
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    get_scheduler,
    AutoModel,
    AutoTokenizer,
    BertTokenizerFast,
    BertForTokenClassification,
    AutoModelForSequenceClassification,
)
from typing import Callable, Union
from utils import logger

logger = logger()


class felflarebert(LightningModule):
    """felflare bert punctuation restoration model"""
    
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
        model: str = 'felflare/bert-restore-punctuation',
        num_workers: int = 4,
        num_labels: int = 2, # number of labels to classify
        # label_names: list[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # For classification, need to know the number of labels .e.g 2 if output is positive or negative
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        
        # self.encoder = AutoModel.from_pretrained(model)
        # hidden = self.encoder.config.hidden_size
        # self.loss_fn = nn.CrossEntropyLoss()
        # self.classifier = nn.Linear(hidden, self.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model,
            num_labels = self.num_labels,
            ignore_mismatched_sizes = True,
        ) 

        self.outputs = []
        self.training_data = None

        # if label_names is None:
        #     self.label_names = [str(i) for i in range(num_labels)]
        # else:
        #     self.label_names = label_names
        # self.id2label = {i: label for i, label in enumerate(self.label_names)}
        # self.label2id = {label: i for i, label in enumerate(self.label_names)}
        # self.encoder.config.id2label = self.id2label
        # self.encoder.config.label2id = self.label2id
        
    
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
        # outputs = self.encoder(
        #     input_ids = input_ids,
        #     attention_mask = attention_mask,
        # )
        # sequence_output = outputs.last_hidden_state[:, 0]

        # logits = self.classifier(sequence_output)
        # loss = None
        # if labels is not None:
        #     loss = self.loss_fn(logits, labels)
        
        # return {'loss': loss, 'logits': logits}
        return self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels,
        )
        
    
    # def decoder(
    #     self,
    #     skip_special_tokens: bool = True,
    #     clean_up_tokenization_spaces: bool = False,
    # ) -> Callable[[str], str]:
    #     def decode(ids):
    #         return self.tokenizer.decode(
    #             ids,
    #             skip_special_tokens = skip_special_tokens,
    #             clean_up_tokenization_spaces = clean_up_tokenization_spaces
    #         ).strip()
        
    #     return decode
    
    def on_test_epoch_end(self):
        logger.info('Test epoch ended')
    
    def store_data(self, d: IntTrainData):
        """Store a TrainData with a model instance for dataloader methods"""
        self.training_data = d
    
    def _verify_data_stored(self):
        if self.training_data is None:
            raise Exception('This model has no stored TrainData data. Call .store_data() before using dataloaders!')
        
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
        outputs = self(
            input_ids = batch['input_ids'],
            attention_mask = batch['attention_mask'],
            labels = labels
        )
        # loss = outputs["loss"]
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

        outputs = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels
        )
        
        # logits = outputs['logits']
        logits = outputs.logits
        predictions = torch.argmax(logits, dim = -1)
        
        self.outputs.append({
            'predictions': predictions.detach().cpu().tolist(),
            'targets': labels.detach().cpu().tolist(),
        })
        
    def save(self, path: str):
        """Save model parameters to path"""
        torch.save(self.state_dict(), path)
        logger.info(f'Saved model to {path}')
    
    def _generate(self, input_dataloader):
        id2label = self.model.config.id2label
        texts, outputs, targets = [], [], []
        with torch.no_grad():
            for batch in tqdm(input_dataloader):
                input_ids = batch['input_ids'].to("cuda")
                attention_mask = batch['attention_mask'].to("cuda")
                labels = batch['labels'].to("cuda")

                batch_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                batch_texts = [text.strip() for text in batch_texts]
                # batch['text'] invalid key

                model_outputs = self(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    labels = labels
                )
                
                logits = model_outputs.logits
                predictions = torch.argmax(logits, dim = -1)
                
                predictions = [id2label[pred.item()] for pred in predictions]

                mytargets = [id2label[label.item()] for label in labels]

                texts.extend(batch_texts)
                outputs.extend(predictions)
                targets.extend(mytargets)
        return texts, outputs, targets
    
    def generate(self, input_dataloader: DataLoader):
        # tuple[list[str], list[str], list[str]]
        self.model.eval()
        return self.to('cuda')._generate(input_dataloader)