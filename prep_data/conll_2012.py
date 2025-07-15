from datasets import load_dataset

class CoNLL2012:

    def __init__(self, split='train'):
        self.data = load_dataset(
            'ontonotes/conll2012_ontonotesv5',
            'english_v4',
            split=split,
            trust_remote_code=True,
            # streaming=True, # doesn't work since have to index 'sentences'
        )

    def features(self):
        return self.data.features

    def dump_json(self, path='conll_2012.jsonl'):
        count = 0
        with open(path, 'w', encoding='utf-8') as file:
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
                    file.write(f'{{"source": "{source}", "target": "{target.rstrip()}"}}\n')
                    count += 1
        print(f"[INFO] Wrote {count} examples to {path}")

if __name__ == '__main__':
    ds = CoNLL2012()
    ds.dump_json()
