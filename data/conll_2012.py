import re

from modules import PrepData


def remove_bio_prefixes(label: str) -> str:
    if len(label) < 2 or label[1] != '-':
        return label
    if label[0] not in ['B', 'I', 'O']:
        raise ValueError(f"Haven't seen {label}")
    return label[2:]

class CoNLL2012(PrepData):

    def __init__(self, split='train'):
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
        out: list[tuple[str, dict[str, dict[str, int]]]] = []
        total_labels: int = 0
        for verb_frame in re.findall(r'\S+ \(.*?\)', s.strip(' ,')):
            front, back = verb_frame.split(' ', 1)
            verb = front.rstrip(' ')
            verb_dict = {}
            related_words = back.lstrip(' (').rstrip(' )')
            for role_label_members in related_words.split(','):
                label, members = role_label_members.strip(' ').split(':')
                label, members = label.strip(' '), members.strip(' ').split(' ')
                verb_dict.setdefault(label, {})
                for member in members:
                    member = member.strip(' ')
                    verb_dict[label].setdefault(member, 0)
                    verb_dict[label][member] += 1
                    total_labels += 1
            out.append((verb, verb_dict))
        return out, total_labels
        

if __name__ == '__main__':
    # ds = CoNLL2012()
    # ds.to_json('conll-2012-srl-512t')
    a, _ = CoNLL2012.unserialize('eat (A: ham burger, B: chicken), drink (C: coke sprite beer sprite)')
    b, _ = CoNLL2012.unserialize('eat (B: chicken, A: ham burger), drink (C: sprite sprite beer coke)')
    print(a)
    print(b)
    print(a == b)
