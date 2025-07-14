from datasets import load_dataset
import json

class CoNLL2012:

    def __init__(self, split='train'):
        self.data = load_dataset('ontonotes/conll2012_ontonotesv5', 'english_v4', split=split)

    def tags(self, feature='srl_frames'):
        return self.data.features[feature].feature.names

    def label(self, i, feature='srl_frames'):
        return self.tags(feature)[i]

    def dump_json(self, path='conll_2012.jsonl'):
        count = 0
        with open(path, 'w', encoding='utf-8') as file:
            for row in self.data:
                source = ' '.join(row['tokens'])
                target = ' '.join([self.label(i) for i in row['pos_tags']])
                json.dump({"source": source, "target": target}, file, ensure_ascii=False)
                file.write("\n")
                count += 1
        print(f"[INFO] Wrote {count} rows to {path}")

if __name__ == '__main__':
    ds = CoNLL2012()
    print(ds.data.features)