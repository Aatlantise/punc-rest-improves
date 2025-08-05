from data.modules import PrepData
from utils import logger, pp

logger = logger()


class CoNLL2000(PrepData):

    def __init__(self, split = 'train'):
        """Loads dataset form hugging face"""
        super().__init__(
            path = 'haeunkim/spacy-conll2000-pos',
            split = split,
            streaming = True,
        )
    
    def id_to_pos_tag(self, i: int) -> str:
        return self.data.features['pos_tags'].feature.names[i]
    
    def src_tgt_pairs(self, task: str):
        if task not in ['pos']:
            raise NotImplementedError(f'Task {task} not implemented. ')
        for example in self.data:
            tokens, tags = example['tokens'], example['pos_tags']
            source = ' '.join(tokens)
            target = ' '.join(map(self.id_to_pos_tag, tags))
            yield source, target

if __name__ == '__main__':
    o = CoNLL2000()
    o.to_json('pos', 'conll-2000-pos')
    
