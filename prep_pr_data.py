from datasets import load_dataset
import re
import random
import json
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

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
    return ''.join(ch for ch in text if ch not in PUNCTUATION_TO_REMOVE)

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

if __name__ == "__main__":
    generate_punctuation_restoration_data()
