import json
import os

from data.modules import PrepData
from tasks.ner import process
from utils import logger, progress

logger = logger(__name__)


class Wikiann(PrepData):
    
    def __init__(self, lang = 'fr', local_dir = None):
        """Loads dataset form hugging face"""
        if local_dir:
            super().__init__(hf_dataset = False)
            splits = ['train', 'dev', 'test']
            for split in splits:
                path = os.path.join(local_dir, lang, f'{split}.jsonl')
                with open(path, 'r') as f:
                    for line in progress(f, 'Wikiann %s %s' % (lang, split)):
                        self.data.append(json.loads(line))
        else:
            super().__init__('tner/wikiann', lang)
        
    @staticmethod
    def id_to_label(id):
        labels = ['B-LOC', 'B-ORG', 'B-PER', 'I-LOC', 'I-ORG', 'I-PER', 'O']
        return labels[id]
    
    def src_tgt_pairs(self, task):
        if task not in ['ner']:
            raise NotImplementedError(f'Task {task} not implemented. ')
        for example in self:
            tokens = example['tokens']
            tokens = list(map(lambda t: '"' if t in ["''", "``"] else t, tokens))
            if len(tokens) < 10:
                continue
            tags = list(map(self.id_to_label, example['tags']))
            yield process(tokens, tags)


if __name__ == '__main__':
    o = Wikiann(local_dir = '../External Datasets/wikiann')
    o.to_json('ner', 'wikiann.fr-ner')