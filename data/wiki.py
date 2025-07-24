import nltk
import re

from data.modules import PrepData
from nltk.tokenize import sent_tokenize, word_tokenize


# Constants
PUNCTUATION_TO_REMOVE = {',', '.', '!', '?', '"', '’', '“', '”', "'"}
MAX_WORDS = 150
MAX_EXCERPTS = 450000


def normalize_text(text):
    """Lowercase and remove specific punctuation and capitalization."""
    text = text.lower()
    return ''.join(ch for ch in text if ch not in PUNCTUATION_TO_REMOVE)


def remove_reference_tags(text):
    """Remove unwanted artifacts like reference tags"""
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\n+', ' ', text).strip()
    return text


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


class WikiPR(PrepData):
    """English Wikipedia prepped for punctuation restoration"""

    def __init__(self):
        super().__init__(
            path='wikimedia/wikipedia',
            name='20231101.en',
            split='train',
            streaming=True
        )

    def src_tgt_pairs(self):
        excerpt_count = 0
        for article in self.data:
            text = article.get("text", "")
            if not text or len(text) < 200:
                continue
            text = remove_reference_tags(text)
            text = sent_tokenize(text)
            for chunk in chunk_sentences(text, max_words=MAX_WORDS):
                target = chunk.strip()
                yield normalize_text(target), target
                excerpt_count += 1
                if excerpt_count >= MAX_EXCERPTS:
                    return


if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('punkt_tab')
    ds = WikiPR()
    ds.to_json('wiki-20231101.en-pr')
