from data.modules import PrepData
from utils import logger

logger = logger()


class CoNLL2003(PrepData):

    def __init__(self, split = 'train'):
        """Loads dataset form hugging face"""
        super().__init__(
            path = 'lhoestq/conll2003',
            split = split,
            streaming = True,
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

    def src_tgt_pairs(self, task: str):
        if task not in ['pos']:
            raise NotImplementedError(f'Task {task} not implemented. ')
        for example in self.data:
            tokens, tags = example['tokens'], example['pos_tags']
            source = ' '.join(tokens)
            target = ' '.join(map(self.id_to_pos_tag, tags))
            yield source, target

if __name__ == '__main__':
    o = CoNLL2003()
    o.to_json('pos', 'conll-2003-pos')
    
