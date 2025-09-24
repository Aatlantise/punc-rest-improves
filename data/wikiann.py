from data.modules import PrepData
from tasks.ner import process
from utils import logger

logger = logger(__name__)


KEEP_PROMPT = False


class WikiAnn(PrepData):

    def __init__(self):
        """Loads dataset form hugging face"""
        super().__init__(path = 'tner/wikiann')
    
    def src_tgt_pairs(self, task: str):
        if task not in ['ner']:
            raise NotImplementedError(f'Task {task} not implemented. ')
        for _, split in self.data.items():
            for example in split:
                tokens = example['tokens']
                tags = map(self.id_to_ner_tag, example['ner_tags'])
                yield process(tokens, tags)


if __name__ == '__main__':
    o = WikiAnn()
    print(o.data.features)