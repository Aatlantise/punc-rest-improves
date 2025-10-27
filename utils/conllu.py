# Reads conllu files
from utils import logger

logger = logger(__name__)


class CoNLLU_Word:
    """See https://universaldependencies.org/format.html
    
    ID: Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens; may be a decimal number for empty nodes (decimal numbers can be lower than 1 but must be greater than 0).
    FORM: Word form or punctuation symbol.
    LEMMA: Lemma or stem of word form.
    UPOS: Universal part-of-speech tag.
    XPOS: Optional language-specific (or treebank-specific) part-of-speech / morphological tag; underscore if not available.
    FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
    HEAD: Head of the current word, which is either a value of ID or zero (0).
    DEPREL: Universal dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
    DEPS: Enhanced dependency graph in the form of a list of head-deprel pairs.
    MISC: Any other annotation.
    """
    
    def __init__(self, line: str):
        components = list(map(
            lambda x: x if x != '_' else None,
            line.split('\t')
        ))
        try:
            if '-' in components[0]:
                a, b = components[0].split('-', 1)
                self.id = (a, b)
            else:
                self.id = int(components[0])
            self.form: str = components[1]
            self.lemma: str = components[2]
            self.upos: str = components[3]
            self.xpos: str = components[4]
            self.feats: dict[str, str] = {
                pair_str.split('=')[0]: pair_str.split('=')[1]
                for pair_str in components[5].split('|')
            } if components[5] else {}
            self.head = int(components[6]) if components[6] else None
            self.deprel: str = components[7]
            self.deps: dict[str, str] = {
                pair_str.split(':')[0]: pair_str.split(':')[1]
                for pair_str in components[8].split('|')
            } if components[8] else {}
            self.misc: dict[str, str] = {
                pair_str.split('=')[0]: pair_str.split('=')[1]
                for pair_str in components[9].split('|')
            } if components[9] else {}
        except Exception as e:
            print('Cannot parse line:')
            print(line)
            raise e


class CoNLLU_Sentence:
    
    def __init__(self, lines: list[str]):
        assert lines[0].startswith('# sent_id')
        self.sent_id = lines[0].split('=', 1)[1].strip()
        
        assert lines[1].startswith('# text')
        self.text = lines[1].split('=', 1)[1].strip()
        
        self.words = list(map(CoNLLU_Word, lines[2:]))
    
    def pos_str(self) -> str:
        """String containing space-separated POS tags of the current sentence"""
        tags = [w.upos for w in self.words if w.upos]
        return ' '.join(tags)


class CoNLLU_File:
    
    def __init__(self, filename):
        lines = open(filename).readlines()
        
        self.entries = []
        accumulator = []
        for line in lines:
            if line.startswith('# global.columns'):
                continue
            elif line == '\n':
                s = CoNLLU_Sentence(accumulator)
                self.entries.append(s)
                accumulator.clear()
            else:
                accumulator.append(line.rstrip())
            
