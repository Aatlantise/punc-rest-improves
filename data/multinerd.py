from data.modules import PrepData
from utils import logger

logger = logger(__name__)



class MultiNERD(PrepData):

    def __init__(self, lang = 'fr'):
        """Loads dataset form hugging face"""
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
    
    def src_tgt_pairs(self, task: str):
        if task not in ['ner']:
            raise NotImplementedError(f'Task {task} not implemented. ')
        for _, split in self.data.items():
            for example in split:
                tokens = example['tokens']
                tags = list(map(self.id_to_label, example['tags']))
                yield ' '.join(tokens), ' '.join(tags)


if __name__ == '__main__':
    o = MultiNERD()
    o.to_json('multinerd.fr-ner')