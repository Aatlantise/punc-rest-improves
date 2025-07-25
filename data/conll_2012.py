import logging
import re
import sys

from data.modules import PrepData


logging.basicConfig(
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level = logging.DEBUG,
    stream = sys.stdout
)
logger = logging.getLogger(__name__)

def remove_bio_prefixes(label: str) -> str:
    if len(label) < 2 or label[1] != '-':
        return label
    if label[0] not in ['B', 'I', 'O']:
        raise ValueError(f"Haven't seen {label}")
    return label[2:]

class CoNLL2012(PrepData):

    def __init__(self, split = 'train'):
        """Loads dataset form hugging face"""
        super().__init__(
            path='ontonotes/conll2012_ontonotesv5',
            name='english_v4',
            split=split,
            # streaming=True, # doesn't work since have to index 'sentences'
        )

    def src_tgt_pairs(self):
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
    
    @staticmethod
    def unserialize(s: str) -> tuple[dict[str, dict[str, dict[str, int]]], int]:
        """
        Given a sequence, divide it into verb frames,
        then for each verb, list the semantic roles and words in the same sentence with that role in a multiset,
        along with the total number of semantic role labels,

        For example: the string 'eat (A: ham burger, B: chicken), drink (C: coke sprite beer sprite)' will return
        {
            'eat_1': {
                'A': {'ham': 1, 'burger': 1},
                'B': {'chicken': 1},
            },
            'drink_1': {
                'C': {'coke': 1, 'sprite': 2, 'beer': 1},
            },
        },
        7
        """
        out: dict[str, dict[str, dict[str, int]]] = {}
        verb_counter: dict[str, int] = {}
        total_labels: int = 0
        for verb_frame in re.finditer(r'(\S+) \((.*?\))', s.strip(' ,')):
            verb, bracket_content = verb_frame.group(1), verb_frame.group(2)
            verb_counter.setdefault(verb, 0)
            verb_counter[verb] += 1
            
            verb = f'{verb}_{verb_counter[verb]}'
            out.setdefault(verb, {})
            verb_dict = out[verb]
            for label_friends in re.finditer(r'([A-Z\-\d]+): (.*?\S)[,)]', bracket_content):
                label, friends = label_friends.group(1), label_friends.group(2)
                verb_dict.setdefault(label, {})
                for friend in friends.split():
                    friend = friend.strip(' ')
                    verb_dict[label].setdefault(friend, 0)
                    verb_dict[label][friend] += 1
                    total_labels += 1
            
        return out, total_labels
        

if __name__ == '__main__':
    o = CoNLL2012.unserialize("know (ARG0: You, ARGM-NEG: n't, V: know, ARG1: that, ARGM-ADV: going into a job), going (V: going, ARG2: into a job)")
    print(o)
    
