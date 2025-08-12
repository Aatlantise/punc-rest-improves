from data.modules import PrepData
from tasks.ner import process
from utils import logger

logger = logger(__name__)


class GENIA(PrepData):

    def __init__(self):
        """Loads dataset form hugging face"""
        super().__init__(path = 'chufangao/GENIA-NER', streaming = True)
    
    @staticmethod
    def id_to_ner_tag(i: int) -> str:
        ner_tags = ['O', 'B-DNA', 'I-DNA', 'B-RNA', 'I-RNA', 'B-cell line', 'I-cell line', 'B-cell type', 'I-cell type', 'B-protein', 'I-protein']
        return ner_tags[i]
    
    def src_tgt_pairs(self, task: str):
        if task not in ['ner']:
            raise NotImplementedError(f'Task {task} not implemented. ')
        for _, split in self.data.items():
            for example in split:
                tokens = example['tokens']
                tags = map(self.id_to_ner_tag, example['ner_tags'])
                yield process(tokens, tags)


if __name__ == '__main__':
    o = GENIA()
    o.to_json('ner', 'genia-ner')