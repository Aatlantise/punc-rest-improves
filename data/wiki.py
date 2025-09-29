import nltk
import random
import re

from argparse import ArgumentParser
from data.modules import PrepData
from nltk.tokenize import sent_tokenize, word_tokenize
from utils import progress, logger

logger = logger(__name__)


# Constants
PUNCTUATION_TO_REMOVE = {',', '.', '!', '?', '"', '’', '“', '”', "'"}
MAX_WORDS = 150
MAX_EXCERPTS = 450000


def bookending_chars(s: str) -> tuple[str, str]:
    """Get first and last non-space characters"""
    stripped = s.strip()
    if len(stripped) < 1:
        return '', ''
    return stripped[0], stripped[-1]

def is_like_citation(s: str) -> bool:
    return re.match(r'\s[A-Za-z\d.\u2013]+', s) is not None # u2013 is the en-dash

def is_like_sentence(s: str) -> bool:
    """Long-ish, starts with a letter and has an ending punctuation"""
    if len(s) <= 30:
        return False
    first, last = bookending_chars(s)
    return first.isalpha() and last in set(',.;:!?') and not is_like_citation(s)

def remove_empty_brackets(s: str) -> str:
    """Removes brackets that have nothing between them
    
    These may originate from parenthesized hyperlinks, the links and their text labels were lost in processing
    """
    s = re.sub(r'\(\W*?\)', '', s)
    s = re.sub(r'\[\W*?\]', '', s)
    return s

def remove_multiple_spaces(s: str) -> str:
    s = re.sub(r' +', ' ', s)
    return s

def remove_repeating_punctuations(s: str) -> str:
    s = re.sub(r'([,;\.] ?){2,}', '\1', s)
    return s

def remove_brackets_with_missing_content(s: str) -> str:
    """Removes brackets that don't neighbour a non-space inside
    
    These might be from missing widgets, like phonetic descriptions and links to recordings
    """
    s = re.sub(r'\(\W+.+?\)', '', s)
    s = re.sub(r'\(.+?\W+\)', '', s)
    s = re.sub(r'\[\W+.+?\]', '', s)
    s = re.sub(r'\[.+?\W+\]', '', s)
    return s

def normalize_spaces(s: str) -> str:
    """Turn non-space separating characters into regular spaces, might be more"""
    s = re.sub(r'\x01', ' ', s)
    s = re.sub(r'\u00a0', ' ', s)
    return s

def pre_process(s: str) -> str:
    """This is done before list-collation"""
    s = remove_empty_brackets(s)
    s = remove_repeating_punctuations(s)
    s = remove_brackets_with_missing_content(s)
    return s

def post_process(s: str) -> str:
    """This is done after list-collation"""
    s = remove_empty_brackets(s)
    s = remove_repeating_punctuations(s)
    s = normalize_spaces(s)
    s = remove_multiple_spaces(s)
    return s

def cleaned(lines: list[str]) -> list[str]:
    """Cleans wikipedia text"""
    picked_lines = []
    for long_lines in lines:
        for l in long_lines.split('\n'):
            if re.match(r'\s[,.;]\s', l):
                continue  # bad sentence, missing words
            elif is_like_sentence(l):
                picked_lines.append(l)
    
    last_sentence = ''
    cleaned_lines = []
    for l in picked_lines:
        l = pre_process(l)
        first, _ = bookending_chars(l)
        if first == '':
            continue
        elif last_sentence == '':
            last_sentence = l
        elif first.islower():
            last_sentence += ' ' + l.lstrip(' \t')
        elif l[-1] == ':' or ' . ' in l or ' : ' in l:  # incomplete list or citation-like
            continue
        else:
            final = post_process(last_sentence).strip()
            cleaned_lines.append(final + '\n')
            last_sentence = l
            
    logger.debug('Cleaning lines... Batch has %d -> 1st round %d -> 2nd round %d' % (len(lines), len(picked_lines), len(cleaned_lines)))
    return cleaned_lines


def normalize_text(text):
    """Lowercase and remove specific punctuation and capitalization."""
    text = text.lower()
    return ''.join(ch for ch in text if ch not in PUNCTUATION_TO_REMOVE)


def mask_text(
    text: str,
    p_mask: float = 0.15,
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
    """English Wikipedia"""

    def __init__(self, lang = 'en'):
        super().__init__(
            path = 'wikimedia/wikipedia',
            name = '20231101.' + lang,
        )
    
    def rations(self, size: int = 5000):
        """Yields `size` lines from the source each time"""
        ration = []
        for _, split in self.data.items():
            for article in progress(split, 'Wikipedia content'):
                text = article.get('text', '')
                if not text or len(text) < 200:
                    continue
                for line in text.split('\n'):
                    ration.append(line)
                    if len(ration) >= size:
                        yield cleaned(ration)
                        ration.clear()
        return

    def src_tgt_pairs(self, task):
        excerpt_count = 0
        for ration in self.rations():
            line = sent_tokenize(''.join(ration))
            for chunk in chunk_sentences(line, max_words = MAX_WORDS):
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
    parser = ArgumentParser()
    parser.add_argument(
        '-l',
        type = str,
        help = 'Language',
    )
    args = parser.parse_args()
    lang = args.l
    
    # nltk.download('punkt')
    # nltk.download('punkt_tab')
    random.seed(42)
    ds = Wiki2023(lang = lang)
    ds.to_json('pr', f'wiki-20231101.{lang}-pr')
    ds.to_json('mlm', f'wiki-20231101.{lang}-mlm')
    
