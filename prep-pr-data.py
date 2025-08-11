from datasets import load_dataset
import re
import random
import json
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import argparse

nltk.download('punkt')
nltk.download('punkt_tab')

# Constants
PUNCTUATION_TO_REMOVE = {',', '.', '!', '?', '"', '’', '“', '”', "'"}
MAX_WORDS = 150
MAX_EXCERPTS = 450000
OUTPUT_PATH = "punctuation_restoration_dataset.jsonl"

def normalize_text(text):
    """Lowercase and remove specific punctuation and capitalization."""
    text = text.lower()
    result = ''.join(ch for ch in text if ch not in PUNCTUATION_TO_REMOVE)
    result = re.sub('\s+', ' ', result)
    return result

def chunk_sentences(sentences, max_words=MAX_WORDS):
    """Split sentences into non-overlapping chunks under max_words."""
    chunks = []
    chunk = []
    word_count = 0

    for sentence in sentences:
        words = word_tokenize(sentence)
        if word_count + len(words) <= max_words:
            chunk.append(sentence)
            word_count += len(words)
        else:
            if chunk:
                chunks.append(' '.join(chunk))
            chunk = [sentence]
            word_count = len(words)

    if chunk:
        chunks.append(' '.join(chunk))

    return chunks

def generate_punctuation_restoration_data():
    # Stream Wikipedia dataset (doesn't download entire corpus)
    wiki_stream = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

    excerpt_count = 0
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as fout:
        for article in wiki_stream:
            text = article.get("text", "")
            if not text or len(text) < 200:
                continue

            # Remove unwanted artifacts like reference tags
            text = re.sub(r'\[\d+\]', '', text)
            text = re.sub(r'\n+', ' ', text).strip()
            

            sentences = sent_tokenize(text)
            chunks = chunk_sentences(sentences, max_words=MAX_WORDS)

            for chunk in chunks:
                target = chunk.strip()
                source = normalize_text(target)
                json.dump({'source': source, 'target': target}, fout, ensure_ascii=False)
                fout.write('\n')

                excerpt_count += 1
                if excerpt_count >= MAX_EXCERPTS:
                    print(f"✅ Done: {excerpt_count} excerpts written to {OUTPUT_PATH}")
                    return

    print(f"⚠️ Only {excerpt_count} excerpts found (less than {MAX_EXCERPTS})")

def generate_conll03_data(OUTPUT_PATH):
    def generate_ner(OUTPUT_PATH, data):
        label_names = data.features['ner_tags'].feature.names
        excerpt_count = 0
        with open (OUTPUT_PATH, 'w', encoding='utf-8') as fout:
            for example in data:
                text = example['tokens']
                # each example is of form: {'id': '2', 'tokens': ['BRUSSELS', '1996-08-22'], 'pos_tags': [22, 11], 'chunk_tags': [11, 12], 'ner_tags': [5, 0]}
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
                        current = f"({token}:{tag[2:]})"
                    elif tag.startswith("I-"):
                        # if it starts with I- then word is inside a phrase
                        current = current[:-len(tag[2:])-2] + f" {token}:{tag[2:]})" # 2 for the I- and 1 for the ":" and 1 for the ")"
                    else:
                        if current:
                            target.append(current)
                        current = ""
                if current:
                    target.append(current)
                target_string = " ".join(target) if target else "O" # O for no prediction
                
                json.dump({'source': source, 'target': target_string}, fout, ensure_ascii=False)
                fout.write('\n')

                excerpt_count += 1
        print(f"prepared from {excerpt_count} excerpts from {OUTPUT_PATH} dataset")
    train = load_dataset("conll2003", split="train", trust_remote_code=True)
    val = load_dataset("conll2003", split="validation", trust_remote_code=True)
    test = load_dataset("conll2003", split="test", trust_remote_code=True)
    
    generate_ner(OUTPUT_PATH+"_train.jsonl", train)
    generate_ner(OUTPUT_PATH+"_val.jsonl", val)
    generate_ner(OUTPUT_PATH+"_test.jsonl", test)
    print("generated three splits")

def generate_conll04_data(OUTPUT_PATH):
    train = load_dataset("DFKI-SLT/conll04", split="train", trust_remote_code=True)
    val = load_dataset("DFKI-SLT/conll04", split="validation", trust_remote_code=True)
    test = load_dataset("DFKI-SLT/conll04", split="test", trust_remote_code=True)
    def gen_RE(OUTPUT_PATH, data):
        excerpt_count = 0
        with open (OUTPUT_PATH, 'w', encoding = 'utf-8') as fout:
            for example in data:
                entities = example['entities']
                text = example['tokens']
                relations = example['relations']
                source = ' '.join(text)
                target = []
                current = ""
                for relation in relations:
                    entity_start_index = relation['head']
                    entity_end_index = relation['tail']
                    start, end = entities[entity_start_index], entities[entity_end_index]

                    start_tokens = ' '.join(text[start['start']:start['end']]) # not start['end']+1 since data accounts for it
                    end_tokens = ' '.join(text[end['start']:end['end']])
                    current = "(" + start_tokens + " " + relation['type'] + " " + end_tokens + ")"
                    target.append(current)

                excerpt_count +=1
                target_string = " ".join(target) if target else "O"
                
                json.dump({'source': source, 'target': target_string}, fout, ensure_ascii=False)
                fout.write('\n')

                excerpt_count += 1
        print(f"prepared from {excerpt_count} excerpts from {OUTPUT_PATH} dataset")
    gen_RE(OUTPUT_PATH+"_train.jsonl", train)
    gen_RE(OUTPUT_PATH+"_val.jsonl", val)
    gen_RE(OUTPUT_PATH+"_test.jsonl", test)
    print("generated three splits")

def generate_TACRED_data(OUTPUT_PATH):
    TACRED_data = load_dataset("DFKI-SLT/tacred", split="train", trust_remote_code=True)
    excerpt_count = 0
    with open (OUTPUT_PATH, 'w', encoding = 'utf-8') as fout:
        for example in TACRED_data:
            # print(example)
            print(len(example['relations']))
            if len(example['relations']) > 0:
                print(len(example['relations']))
                excerpt_count += 1
    print(f"prepared from {excerpt_count} excerpts from conll04 dataset")
            

def generate_GENIA_data(OUTPUT_PATH):
    train = load_dataset("chufangao/GENIA-NER", split="train")
    val = load_dataset("chufangao/GENIA-NER", split="validation")
    test = load_dataset("chufangao/GENIA-NER", split="test")
    
    def gen_RE(OUTPUT_PATH, data):
        tag_names = data.features['ner_tags'].feature.names
        excerpt_count = 0
        with open (OUTPUT_PATH, 'w', encoding = 'utf-8') as fout:
            for example in data:
                tokens = example['tokens']
                tags = [tag_names[tag] for tag in example['ner_tags']]
                source = ' '.join(tokens)
                curr = ""
                target = []
                for token, tag in zip(tokens, tags):
                    if tag.startswith("B-"):
                        if curr:
                            target.append(curr)
                        curr =  f"({token}:{tag[2:]})"
                    elif tag.startswith("I-"):
                        curr = curr[:-len(tag[2:])-2] + f" {token}:{tag[2:]})"
                    else:
                        if curr:
                            target.append(curr)
                        curr = ""
                if curr:
                    target.append(curr)
                target_string = " ".join(target) if target else "O" # O if no predictions
                    
                json.dump({'source': source, 'target': target_string}, fout, ensure_ascii=False)
                fout.write('\n')
                excerpt_count += 1
        print(f"prepared from {excerpt_count} excerpts from {OUTPUT_PATH} dataset")
    gen_RE(OUTPUT_PATH+"_train.jsonl", train)
    gen_RE(OUTPUT_PATH+"_val.jsonl", val)
    gen_RE(OUTPUT_PATH+"_test.jsonl", test)
    print("generated three splits")


def generate_ontonotes_data(OUTPUT_PATH):
    train = load_dataset("tner/ontonotes5", split="train")
    val = load_dataset("tner/ontonotes5", split="validation")
    test = load_dataset("tner/ontonotes5", split="test")
    # ontonotes_data = load_dataset("ontonotes/conll2012_ontonotesv5", 'english_v4', split="train", trust_remote_code=True) # this has a non-matching split error, can't be used
    # from https://huggingface.co/datasets/tner/ontonotes5
    label_map = {
        "O": 0,
        "B-CARDINAL": 1,
        "B-DATE": 2,
        "I-DATE": 3,
        "B-PERSON": 4,
        "I-PERSON": 5,
        "B-NORP": 6,
        "B-GPE": 7,
        "I-GPE": 8,
        "B-LAW": 9,
        "I-LAW": 10,
        "B-ORG": 11,
        "I-ORG": 12, 
        "B-PERCENT": 13,
        "I-PERCENT": 14, 
        "B-ORDINAL": 15, 
        "B-MONEY": 16, 
        "I-MONEY": 17, 
        "B-WORK_OF_ART": 18, 
        "I-WORK_OF_ART": 19, 
        "B-FAC": 20, 
        "B-TIME": 21, 
        "I-CARDINAL": 22, 
        "B-LOC": 23, 
        "B-QUANTITY": 24, 
        "I-QUANTITY": 25, 
        "I-NORP": 26, 
        "I-LOC": 27, 
        "B-PRODUCT": 28, 
        "I-TIME": 29, 
        "B-EVENT": 30,
        "I-EVENT": 31,
        "I-FAC": 32,
        "B-LANGUAGE": 33,
        "I-PRODUCT": 34,
        "I-ORDINAL": 35,
        "I-LANGUAGE": 36
    }
    tag_names = list(label_map)

    def gen_ner(OUTPUT_PATH, data):
        excerpt_count = 0
        with open (OUTPUT_PATH, 'w', encoding = 'utf-8') as fout:
            for example in data:
                tokens = example['tokens']
                tags = [tag_names[tag] for tag in example['tags']]
                source = ' '.join(tokens)
                curr = ""
                target = []
                for token, tag in zip(tokens, tags):
                    if tag.startswith("B-"):
                        if curr:
                            target.append(curr)
                        curr =  f"({token}:{tag[2:]})"
                    elif tag.startswith("I-"):
                        curr = curr[:-len(tag[2:])-2] + f" {token}:{tag[2:]})"
                    else:
                        if curr:
                            target.append(curr)
                        curr = ""
                if curr:
                    target.append(curr)
                target_string = " ".join(target) if target else "O" # O if no predictions
                    
                json.dump({'source': source, 'target': target_string}, fout, ensure_ascii=False)
                fout.write('\n')
                excerpt_count += 1
        print(f"prepared from {excerpt_count} excerpts from {OUTPUT_PATH} dataset")
    gen_ner(OUTPUT_PATH+"_train.jsonl", train)
    gen_ner(OUTPUT_PATH+"_val.jsonl", val)
    gen_ner(OUTPUT_PATH+"_test.jsonl", test)
    print("generated three splits")
    
# def generate_conll00_data(OUTPUT_PATH):
#     conl00_data = load_dataset("eriktks/conll2000", split="train", trust_remote_code=True)
#     excerpt_counter = 0
#     tag_names = conl00_data.features['chunk_tags'].feature.names
#     excerpt_count = 0
#     with open (OUTPUT_PATH, 'w', encoding = 'utf-8') as fout:
#         for example in conl00_data:
#             tokens = example['tokens']
#             source = ' '.join(tokens)
#             tags = [tag_names[tag] for tag in example['chunk_tags']]
#             target = []
#             curr=""
#             for token, tag in zip(tokens, tags):
#                 if tag.startswith("B-"):
#                     if curr:
#                         target.append(curr)
#                     curr =  f"({token}:{tag[2:]})"
#                 elif tag.startswith("I-"):
#                     curr = curr[:-len(tag[2:])-2] + f" {token}:{tag[2:]})"
#                 else:
#                     if curr:
#                         target.append(curr)
#                     curr = ""
#             if curr:
#                 target.append(curr)
#             target_string = " ".join(target) if target else "O" # O if no predictions
#             json.dump({'source': source, 'target': target_string}, fout, ensure_ascii=False)
#             fout.write('\n')
#             excerpt_count += 1
#     print(f"prepared from {excerpt_count} excerpts from conll2000 dataset")

def generate_conll00_data(OUTPUT_PATH):
    data = load_dataset("eriktks/conll2000", split="train", trust_remote_code=True)
    # manually create a validation split
    split = data.train_test_split(test_size = 0.1, seed = 42)
    train = split["train"]
    val = split["test"]
    test = load_dataset("eriktks/conll2000", split="test", trust_remote_code=True)
    def gen_chunking(OUTPUT_PATH, data):
        excerpt_counter = 0
        tag_names = data.features['chunk_tags'].feature.names
        excerpt_count = 0
        with open (OUTPUT_PATH, 'w', encoding = 'utf-8') as fout:
            for example in data:
                tokens = example['tokens']
                source = ' '.join(tokens)
                tags = [tag_names[tag] for tag in example['chunk_tags']]
                
                target_string = " ".join(tags)
                json.dump({'source': source, 'target': target_string}, fout, ensure_ascii=False)
                fout.write('\n')
                excerpt_count += 1
        print(f"prepared from {excerpt_count} excerpts from {OUTPUT_PATH} dataset")
    gen_chunking(OUTPUT_PATH+"_train.jsonl", train)
    gen_chunking(OUTPUT_PATH+"_val.jsonl", val)
    gen_chunking(OUTPUT_PATH+"_test.jsonl", test)
    print("prepared three splits")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="conll2003")
    args = parser.parse_args()
    print(f"preparing {args.dataset_name}")
    if args.dataset_name == "wikipedia":
        generate_punctuation_restoration_data()
    elif args.dataset_name == "conll2003": # for NER
        generate_conll03_data("conll2003")
    elif args.dataset_name == "conll2004": # for RE
        generate_conll04_data("conll2004")
    elif args.dataset_name == "TACRED": # for RE
        generate_TACRED_data("TACRED_data.jsonl")
    elif args.dataset_name == "GENIA": # for NER 
        generate_GENIA_data("GENIA")
    elif args.dataset_name == "ontonotes": # for NER
        generate_ontonotes_data("ontonotes")
    elif args.dataset_name == "conll2000": # for chunking
        generate_conll00_data("conll2000")
    else:
        print(f"missing data for {args.dataset_name}:")
