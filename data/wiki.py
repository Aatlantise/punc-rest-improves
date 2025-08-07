import nltk
import random
import re

from data.modules import PrepData
from nltk.tokenize import sent_tokenize, word_tokenize
from utils import progress

# Constants
PUNCTUATION_TO_REMOVE = {',', '.', '!', '?', '"', '’', '“', '”', "'"}
MAX_WORDS = 150
MAX_EXCERPTS = 450000


def normalize_text(text):
    """Lowercase and remove specific punctuation and capitalization."""
    text = text.lower()
    return ''.join(ch for ch in text if ch not in PUNCTUATION_TO_REMOVE)


def mask_text(
    text: str,
    p_mask: float = 0.15,
    # p_mask_as_token: float = 0.8,
    # p_mask_as_random_word: float = 0.1,
) -> tuple[str, str]:
    """Mask tokens according to parameters and return source and target strings. """
    source_words = word_tokenize(text)
    num_words = len(source_words)
    target_words = []
    mask_indices = random.sample(range(1, num_words - 1), int(num_words * p_mask))
    for i in sorted(mask_indices):
        sentinel = '<extra_id_%d>' % i
        target_words.append(sentinel)
        target_words.append(source_words[i])
        source_words[i] = sentinel
    return ' '.join(source_words), ' '.join(target_words)


def remove_reference_tags(text):
    """Remove unwanted artifacts like reference tags"""
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\n+', ' ', text).strip()
    return text


def chunk_sentences(sentences, max_words = MAX_WORDS) -> list[str]:
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


class Wiki2023(PrepData):
    """English Wikipedia prepped for punctuation restoration"""

    def __init__(self):
        super().__init__(
            path = 'wikimedia/wikipedia',
            name = '20231101.en',
        )

    def src_tgt_pairs(self, task):
        excerpt_count = 0
        for _, split in self.data.items():
            for article in progress(split, 'Wiki dataprep for ' + task):
                text = article.get("text", "")
                if not text or len(text) < 200:
                    continue
                    
                text = remove_reference_tags(text)
                text = sent_tokenize(text)
                for chunk in chunk_sentences(text, max_words = MAX_WORDS):
                    target = chunk.strip()
                    
                    if task == 'pr':
                        yield normalize_text(target), target
                    elif task == 'mlm':
                        yield mask_text(chunk.strip())
                    else:
                        raise NotImplementedError(f'Task {task} not implemented. ')
                    
                    excerpt_count += 1
                    if excerpt_count >= MAX_EXCERPTS:
                        return


if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('punkt_tab')
    random.seed(42)
    ds = Wiki2023()
    ds.to_json('pr', 'wiki-20231101.en-pr')
    ds.to_json('mlm', 'wiki-20231101.en-mlm')
