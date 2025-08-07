import json

from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from typing import Generator
from utils import logger

logger = logger()


class PrepData:
    """Loads dataset with hugging face API, preprocesses it and saves to jsonl
    
    Subclasses should implement `src_tgt_pairs`
    """

    def __init__(self, hf_dataset: bool = True, **kwargs) -> None:
        """Loads dataset form hugging face"""
        if hf_dataset:
            self.data = load_dataset(trust_remote_code = True, **kwargs)
        else:
            self.data = []
        

    def src_tgt_pairs(self, task: str) -> Generator[tuple[str, str], None, None]:
        """A generator function of source-target pairs as examples of training data"""
        pass

    def to_json(self, task: str, name: str = None) -> int:
        """Output data to JSONL

        Default path is `outputs/datasets/` with the jsonl file named after the caller class.
        """
        name = name or self.__class__.__name__ + '-' + task
        path = 'outputs/datasets/' + name + '.jsonl'
        num_lines = 0
        with open(path, 'w', encoding='utf-8') as file:
            for source, target in self.src_tgt_pairs(task):
                json.dump({'source': source, 'target': target}, file, ensure_ascii = False)
                file.write('\n')
                num_lines += 1
        logger.info(f'Wrote {num_lines} lines to {path}')
        return num_lines


class TrainData:
    """Reads dataset from jsonl and provides dataloaders for training"""
    
    def __init__(self, jsonl_path: str):
        """Read and split dataset in JSONL"""
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
    
    def loader(
        self,
        split: str,
        tokenizer,
        max_seq_length: int,
        eval_batch_size: int,
        **kwargs,
    ):
        """Dataloader for data with set tokenizer and other parameters"""
        
        def preprocess(example):
            sources = tokenizer(
                example['source'],
                max_length = max_seq_length,
                truncation = True,
                padding = 'max_length',
            )
            targets = tokenizer(
                example['target'],
                max_length = max_seq_length,
                truncation = True,
                padding = 'max_length',
            )
            sources['labels'] = targets['input_ids']
            return sources
        
        ds = Dataset.from_list(self.data[split]).map(preprocess, batched = True)
        ds.set_format(type = 'torch', columns = ['input_ids', 'attention_mask', 'labels'])
        dl = DataLoader(ds, batch_size = eval_batch_size, **kwargs)
        return dl
    
    