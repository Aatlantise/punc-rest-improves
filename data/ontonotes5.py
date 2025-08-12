from data.modules import PrepData
from tasks.ner import process
from utils import logger

logger = logger(__name__)


class OntoNotes5(PrepData):

    def __init__(self):
        """Loads dataset form hugging face"""
        super().__init__(path = 'tner/ontonotes5', streaming = True)
    
    @staticmethod
    def id_to_ner_tag(i: int) -> str:
        ner_tags = [
            'O', 'B-CARDINAL', 'B-DATE', 'I-DATE', 'B-PERSON', 'I-PERSON',
            'B-NORP', 'B-GPE', 'I-GPE', 'B-LAW', 'I-LAW', 'B-ORG', 'I-ORG',
            'B-PERCENT', 'I-PERCENT', 'B-ORDINAL', 'B-MONEY', 'I-MONEY',
            'B-WORK_OF_ART', 'I-WORK_OF_ART', 'B-FAC', 'B-TIME', 'I-CARDINAL',
            'B-LOC', 'B-QUANTITY', 'I-QUANTITY', 'I-NORP', 'I-LOC', 'B-PRODUCT',
            'I-TIME', 'B-EVENT', 'I-EVENT', 'I-FAC', 'B-LANGUAGE', 'I-PRODUCT',
            'I-ORDINAL', 'I-LANGUAGE'
        ]
        return ner_tags[i]
    
    def src_tgt_pairs(self, task: str):
        if task not in ['ner']:
            raise NotImplementedError(f'Task {task} not implemented. ')
        for _, split in self.data.items():
            for example in split:
                tokens = example['tokens']
                tags = map(self.id_to_ner_tag, example['tags'])
                yield process(tokens, tags)


if __name__ == '__main__':
    o = OntoNotes5()
    o.to_json('ner', 'ontonotes5-ner')