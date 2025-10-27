from data.modules import PrepData
from utils import logger

logger = logger(__name__)

def remove_bio_prefixes(label: str) -> str:
    if len(label) < 2 or label[1] != '-':
        return label
    if label[0] not in ['B', 'I', 'O']:
        raise ValueError(f"Haven't seen {label}")
    return label[2:]

class CoNLL2012(PrepData):

    def __init__(self):
        """Loads dataset form hugging face"""
        super().__init__(
            path = 'ontonotes/conll2012_ontonotesv5',
            name = 'english_v4',
            # streaming=True, # doesn't work since have to index 'sentences'
        )

    def src_tgt_pairs(self, task: str):
        if task not in ['srl']:
            raise NotImplementedError(f'Task {task} not implemented. ')
        for paragraph in self.data['sentences']:
            for sentence in paragraph:
                words: list[str] = sentence['words']
                source = ' '.join(words)
                target = ''
                for srl_frame in sentence['srl_frames']:
                    verb: str = srl_frame['verb']
                    target += '%s (' % verb
                    label_dict: dict[str, list[str]] = {}
                    for i in range(len(words)):
                        label = remove_bio_prefixes(srl_frame['frames'][i])
                        if label == 'O': continue
                        word: str = words[i]
                        label_dict.setdefault(label, []).append(word)
                    for label, words_with_label in label_dict.items():
                        target += label + ': ' + ' '.join(words_with_label) + ', '
                    target = target.rstrip(' ,')
                    target += '), '
                yield source, target.rstrip(' ,')

if __name__ == '__main__':
    o = CoNLL2012()
    o.to_json('srl', 'conll-2012-srl')
    
