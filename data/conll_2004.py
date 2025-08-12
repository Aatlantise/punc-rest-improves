from data.modules import PrepData
from utils import logger

logger = logger(__name__)


class CoNLL2004(PrepData):

    def __init__(self):
        """Loads dataset form hugging face"""
        super().__init__(path = 'DFKI-SLT/conll04', streaming = True)
    
    def src_tgt_pairs(self, task: str):
        if task not in ['re']:
            raise NotImplementedError(f'Task {task} not implemented. ')
        for _, split in self.data.items():
            for example in split:
                entities = example['entities']
                text = example['tokens']
                relations = example['relations']
                source = ' '.join(text)
                target = []
                current = ""
                for relation in relations:
                    entity_start_index = relation['head']
                    entity_end_index = relation['tail']
                    start, end = entities[entity_start_index], entities[entity_end_index]
                    
                    start_tokens = ' '.join(
                        text[start['start']:start['end']]
                    )  # not start['end']+1 since data accounts for it
                    end_tokens = ' '.join(text[end['start']:end['end']])
                    current = "(" + start_tokens + " " + relation['type'] + " " + end_tokens + ")"
                    target.append(current)
                
                target_string = " ".join(target) if target else "O"
                yield source, target_string


if __name__ == '__main__':
    o = CoNLL2004()
    o.to_json('re', 'conll-2004-re')