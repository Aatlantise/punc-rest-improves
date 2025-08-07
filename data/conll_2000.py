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
    
    def id_to_chunk_tag(self, i: int) -> str:
        return self.data['train'].features['chunk_tags'].feature.names[i]
    
    def src_tgt_pairs(self, task: str):
        if task not in ['pos', 'chunking']:
            raise NotImplementedError(f'Task {task} not implemented. ')
        for _, split in self.data.items():
            for example in split:
                tokens = example['tokens']
                source = ' '.join(tokens)
                match task:
                    case 'pos':
                        tags = example['pos_tags']
                        target = ' '.join(map(self.id_to_pos_tag, tags))
                        yield source, target
                    case 'chunking':
                        tags = map(self.id_to_chunk_tag, example['chunk_tags'])
                        target = []
                        curr = ""
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
    o = CoNLL2000()
    o.to_json('pos', 'conll-2000-pos')
    o.to_json('chunking', 'conll-2000-chunking')
    
