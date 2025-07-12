from datasets import load_dataset
import re
import random
import json
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

MAX_WORDS = 150
MAX_EXCERPTS = 450000
OUTPUT_PATH = "processed_conll_data.jsonl"


def generate_conll_data():
    conll_data = load_dataset("conll2003", split="train", trust_remote_code=True)
    label_names = conll_data.features['ner_tags'].feature.names
    excerpt_count = 0

    with open (OUTPUT_PATH, 'w', encoding='utf-8') as fout:
        for example in conll_data:
            text = example['tokens']
            # tags = example['ner_tags']
            tags = [label_names[tag] for tag in example['ner_tags']]
            
            source = " ".join(text)
            target = []
            current = ""

            for token, tag in zip(text, tags):
                if tag.startswith("B-"):
                    # if it starts with B- then this is the start of new phrase
                    if current:
                        target.append(current)
                    current = f"{token}:{tag[2:]}"
                elif tag.startswith("I-"):
                    # if it starts with I- then word is inside a phrase
                    current = current[:-len(tag[2:])] + f" {token}:{tag[2:]}"
                else:
                    if current:
                        target.append(current)
                    current = ""
            if current:
                target.append(current)
            target_string = " ".join(target) if target else "O"

            json.dump({'source': source, 'target': target_string}, fout, ensure_ascii=False)
            fout.write('\n')

            excerpt_count += 1
    print(f"prepared from {excerpt_count} excerpts from conll dataset")

if __name__ == "__main__":
    generate_conll_data()