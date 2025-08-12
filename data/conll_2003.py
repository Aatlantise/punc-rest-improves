from data.modules import PrepData
from tasks.ner import process as ner_process
from utils import logger

logger = logger(__name__)


class CoNLL2003(PrepData):

    def __init__(self):
        """Loads dataset form hugging face"""
        super().__init__(path = 'lhoestq/conll2003')
    
    @staticmethod
    def id_to_pos_tag(i: int) -> str:
         pos_tags = [
             '"', "''", '#', '$', '(', ')', ',', '.', ':', '``',
             'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN',
             'NNP', 'NNPS', 'NNS', 'NN|SYM', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS',
             'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB'
         ]
         return pos_tags[i]
    
    @staticmethod
    def id_to_ner_tag(i: int) -> str:
        ner_tags = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
        return ner_tags[i]

    def src_tgt_pairs(self, task: str):
        if task not in ['pos', 'ner']:
            raise NotImplementedError(f'Task {task} not implemented. ')
        for _, split in self.data.items():
            for example in split:
                tokens = example['tokens']
                match task:
                    case 'pos':
                        source = ' '.join(tokens)
                        target = ' '.join(map(self.id_to_pos_tag, example['pos_tags']))
                        yield source, target
                    case 'ner':
                        tags = map(self.id_to_ner_tag, example['ner_tags'])
                        yield ner_process(tokens, tags)


if __name__ == '__main__':
    o = CoNLL2003()
    o.to_json('pos', 'conll-2003-pos')
    o.to_json('ner', 'conll-2003-ner')