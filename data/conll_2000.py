import logging
import sys

from data.modules import PrepData


logging.basicConfig(
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level = logging.DEBUG,
    stream = sys.stdout
)
logger = logging.getLogger(__name__)


class CoNLL2000(PrepData):

    def __init__(self, split = 'train'):
        """Loads dataset form hugging face"""
        super().__init__(
            path = 'haeunkim/spacy-conll2000-pos',
            split = split,
            streaming = True,
        )

    def src_tgt_pairs(self, task: str):
        if task not in ['pos']:
            raise NotImplementedError(f'Task {task} not implemented. ')
        for example in self.data:
            tokens, spacy_pos_str = example['tokens'], example['spacy_pos_str']
            source = ' '.join(tokens)
            target = ' '.join(spacy_pos_str.split()[1::2]) # only odd indices are pos labels
            yield source, target

if __name__ == '__main__':
    o = CoNLL2000()
    o.to_json('pos', 'conll-2000-pos')
    
