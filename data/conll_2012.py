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
                        if label in label_dict:
                            label_dict[label].append(word)
                        else:
                            label_dict[label] = [word]
                    for label, words_with_label in label_dict.items():
                        target += label + ': ' + ' '.join(words_with_label) + ', '
                    target = target.rstrip(' ,')
                    target += '), '
                yield source, target.rstrip(' ,')


if __name__ == '__main__':
    ds = CoNLL2012()
    ds.to_json('conll-2012-srl-512t')
