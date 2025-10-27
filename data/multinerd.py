import json
import os

from data.modules import PrepData
from tasks.ner import process
from utils import logger

logger = logger(__name__)



class MultiNERD(PrepData):

    def __init__(self, lang = 'fr', local_dir = None):
        """Loads dataset form hugging face"""
        if local_dir:
            super().__init__(hf_dataset = False)
            path = os.path.join(local_dir, f'{lang}.jsonl')
            with open(path, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line))
        else:
            super().__init__('tner/multinerd', lang)
    
    @staticmethod
    def id_to_label(id):
        labels = [
            'O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG',
            'B-ANIM', 'I-ANIM', 'B-BIO', 'I-BIO', 'B-CEL', 'I-CEL', 'B-DIS', 'I-DIS',
            'B-EVE', 'I-EVE', 'B-FOOD', 'I-FOOD', 'B-INST', 'I-INST', 'B-MEDIA',
            'I-MEDIA', 'B-PLANT', 'I-PLANT', 'B-MYTH', 'I-MYTH', 'B-TIME', 'I-TIME',
            'B-VEHI', 'I-VEHI', 'B-SUPER', 'I-SUPER', 'B-PHY', 'I-PHY'
        ]
        return labels[id]
    
    def src_tgt_pairs(self, task):
        if task not in ['ner']:
            raise NotImplementedError(f'Task {task} not implemented. ')
        for example in self:
            tokens = example['tokens']
            tags = list(map(self.id_to_label, example['tags']))
            yield process(tokens, tags)


if __name__ == '__main__':
    o = MultiNERD()
    o.to_json('ner', 'multinerd.fr-ner')