from data.modules import PrepData
from utils import logger

logger = logger()


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
                source = ' '.join(tokens)
                curr = ""
                target = []
                for token, tag in zip(tokens, tags):
                    if tag.startswith("B-"):
                        if curr:
                            target.append(curr)
                        curr = f"({token}:{tag[2:]})"
                    elif tag.startswith("I-"):
                        curr = curr[:-len(tag[2:]) - 2] + f" {token}:{tag[2:]})"
                    else:
                        if curr:
                            target.append(curr)
                        curr = ""
                if curr:
                    target.append(curr)
                target_string = " ".join(target) if target else "O"  # O if no predictions
                yield source, target_string


if __name__ == '__main__':
    o = GENIA()
    o.to_json('ner', 'genia-ner')