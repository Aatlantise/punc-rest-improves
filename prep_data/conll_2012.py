from dataset_mod import DatasetModule


class CoNLL2012(DatasetModule):

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
                words = sentence['words']
                source = ' '.join(words)
                target = ''
                for srl_frame in sentence['srl_frames']:
                    target += f'({srl_frame["verb"]}: '
                    for i in range(len(words)):
                        bio_label = srl_frame['frames'][i]
                        if bio_label != 'O':
                            target += f'[{words[i]}: {bio_label}] '
                    target = target.rstrip() + ') '
                yield source, target.rstrip()


if __name__ == '__main__':
    ds = CoNLL2012()
    ds.to_json('conll-2012-srl')
