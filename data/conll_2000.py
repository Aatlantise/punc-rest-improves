from data.modules import PrepData
from utils import logger, pp

logger = logger()


class CoNLL2000(PrepData):

    def __init__(self):
        """Loads dataset form hugging face"""
        super().__init__(
            path = 'haeunkim/spacy-conll2000-pos',
            streaming = True,
        )
    
    def id_to_pos_tag(self, i: int) -> str:
        return self.data['train'].features['pos_tags'].feature.names[i]
    
    def src_tgt_pairs(self, task: str):
        if task not in ['pos']:
            raise NotImplementedError(f'Task {task} not implemented. ')
        for _, split in self.data.items():
            for example in split:
                tokens, tags = example['tokens'], example['pos_tags']
                source = ' '.join(tokens)
                target = ' '.join(map(self.id_to_pos_tag, tags))
                yield source, target

if __name__ == '__main__':
    o = CoNLL2000()
    print(o.data['train'])
    o.to_json('pos', 'conll-2000-pos')
    
