import logging

from datasets import load_dataset
from json import dump
from typing import Generator

class DatasetModule:

    def __init__(self, **kwargs) -> None:
        """Loads dataset form hugging face"""
        self.data = load_dataset(**kwargs)

    def src_tgt_pairs(self) -> Generator[tuple[str, str], None, None]:
        """A generator function of source-target pairs as examples of training data"""
        pass

    def to_json(self, filename:str=None) -> int:
        """Output data to JSONL

        Default path is `datasets` with the jsonl file named after the caller class.
        """
        filename = filename or self.__class__.__name__
        path = 'datasets/' + filename + '.jsonl'
        num_lines = 0
        with open(path, 'w', encoding='utf-8') as file:
            for source, target in self.src_tgt_pairs():
                dump({'source': source, 'target': target}, file, ensure_ascii=False)
                file.write('\n')
                num_lines += 1
        logging.info(f'Wrote {num_lines} lines to {path}')
        return num_lines
