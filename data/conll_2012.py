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
    def unserialize(s: str) -> tuple[list[tuple[str, dict[str, dict[str, int]]]], int]:
        """
        Given a sequence, divide it into verb frames,
        then for each verb, list the semantic roles and words in the same sentence with that role in a multiset,
        along with the total number of semantic role labels,

        For example: the string 'eat (A: ham burger, B: chicken), drink (C: coke sprite beer sprite)' will return
        {
            'eat': {
                'A': {'ham': 1, 'burger': 1},
                'B': {'chicken': 1},
            },
            'drink': {
                'C': {'coke': 1, 'sprite': 2, 'beer': 1},
            },
        },
        7
        """
        logger.debug('Unserialize: got string')
        logger.debug(s)
        out: list[tuple[str, dict[str, dict[str, int]]]] = []
        total_labels: int = 0
        for verb_frame in re.findall(r'\S+ \(.*?\)', s.strip(' ,')):
            logger.debug('Unserialize: found verb frame')
            logger.debug(verb_frame)
            verb_frame_parts = verb_frame.split(' ', 1)
            front, back = verb_frame_parts[0], verb_frame_parts[1]
            verb = front.rstrip(' ')
            verb_dict = {}
            related_words = back.lstrip(' (').rstrip(' )')
            for role_label_members in related_words.split(','):
                logger.debug('Unserialize: found role label members string')
                logger.debug(role_label_members)
                role_label_members_split = role_label_members.strip(' ').split(':')
                if len(role_label_members_split) < 2:
                    logger.debug('BAD SPLIT')
                    logger.debug(role_label_members_split)
                    continue
                label, members = role_label_members_split[0].strip(' '), role_label_members_split[1].strip(' ').split(' ')
                verb_dict.setdefault(label, {})
                for member in members:
                    member = member.strip(' ')
                    verb_dict[label].setdefault(member, 0)
                    verb_dict[label][member] += 1
                    total_labels += 1
            out.append((verb, verb_dict))
        return out, total_labels
        

if __name__ == '__main__':
    o = CoNLL2012.unserialize('eat (A: ham burger, B: chicken), drink (C: coke sprite beer sprite)')
    print(o)
    