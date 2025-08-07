from data.modules import PrepData
from utils import logger

logger = logger()


class CoNLL2003(PrepData):

    def __init__(self):
        """Loads dataset form hugging face"""
        super().__init__(path = 'lhoestq/conll2003')
    
    @staticmethod
    def id_to_pos_tag(i: int) -> str:
         pos_tags = [
             '"', "''", '#', '$', '(', ')', ',', '.', ':', '``',
             'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN',
             'NNP', 'NNPS', 'NNS', 'NN|SYM', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS',
             'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB'
         ]
         return pos_tags[i]
    
    @staticmethod
    def id_to_ner_tag(i: int) -> str:
        ner_tags = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
        return ner_tags[i]

    def src_tgt_pairs(self, task: str):
        if task not in ['pos', 'ner']:
            raise NotImplementedError(f'Task {task} not implemented. ')
        for _, split in self.data.items():
            for example in split:
                tokens = example['tokens']
                source = ' '.join(tokens)
                match task:
                    case 'pos':
                        target = ' '.join(map(self.id_to_pos_tag, example['pos_tags']))
                        yield source, target
                    case 'ner':
                        tags = map(self.id_to_ner_tag, example['ner_tags'])
                        target = []
                        current = ""
                        for token, tag in zip(tokens, tags):
                            if tag.startswith("B-"):
                                # if it starts with B- then this is the start of new phrase
                                if current:
                                    target.append(current)
                                current = f"({token}:{tag[2:]})"
                            elif tag.startswith("I-"):
                                # if it starts with I- then word is inside a phrase
                                # 2 for the I- and 1 for the ":" and 1 for the ")"
                                current = current[:-len(tag[2:]) - 2] + f" {token}:{tag[2:]})"
                            else:
                                if current:
                                    target.append(current)
                                current = ""
                        if current:
                            target.append(current)
                        yield source, ' '.join(target) if target else "O"  # O for no prediction


if __name__ == '__main__':
    o = CoNLL2003()
    o.to_json('pos', 'conll-2003-pos')
    o.to_json('ner', 'conll-2003-ner')