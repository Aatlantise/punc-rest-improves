import logging
import re
import sys

from data.modules import PrepData


logging.basicConfig(
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level = logging.DEBUG,
    stream = sys.stdout
)
logger = logging.getLogger(__name__)

class CoNLL2003(PrepData):

    def __init__(self, split = 'train'):
        """Loads dataset form hugging face"""
        super().__init__(
            path = 'lhoestq/conll2003',
            split = split,
            # streaming=True, # doesn't work since have to index 'sentences'
        )
    
    @staticmethod
    def id_to_pos_tag(i: int) -> str:
         pos_tags = [
             '"', "''", '#', '$', '(', ')', ',', '.', ':', '``',
             'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN',
             'NNP', 'NNPS', 'NNS', 'NN|SYM', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS',
             'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB'
         ]
         return pos_tags[i]

    def src_tgt_pairs(self, task = 'pos'):
        if task not in ['pos']:
            raise NotImplementedError(f'Task {task} not implemented. ')
        for example in self.data:
            tokens, tags = example['tokens'], example['pos_tags']
            source = ' '.join(tokens)
            target = ''
            for token, tag in zip(tokens, tags):
                if tag < 10:
                    target += token + ' '
                else:
                    target += token + '-' + self.id_to_pos_tag(tag) + ' '
            yield source, target.rstrip(' ')

if __name__ == '__main__':
    o = CoNLL2003()
    o.to_json('conll-2003-pos')
    
