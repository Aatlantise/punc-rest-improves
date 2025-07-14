from datasets import load_dataset
import json

class CoNLL2012:

    def __init__(self, split='train'):
        self.data = load_dataset("ramybaly/conll2012", split=split)
        self.pos_tags = self.data.features['pos_tags'].feature.names

    def label(self, i):
        return self.pos_tags[i]

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
    ds.dump_json()