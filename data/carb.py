import sys

from data.modules import PrepData
from tasks.oie import normalize_quotes
from utils import logger

logger = logger(__name__)


class CaRB(PrepData):

    def __init__(self, repo_path: str):
        """Loads dataset from TSVs
        
        Refer to https://github.com/dair-iitd/CaRB.
        """
        super().__init__(hf_dataset = False)
        with open(f'{repo_path}/data/gold/dev.tsv', 'r') as file:
            for line in file:
                self.data.append(line)
        with open(f'{repo_path}/data/gold/test.tsv', 'r') as file:
            for line in file:
                self.data.append(line)
    
    def src_tgt_pairs(self, task: str):
        if task not in ['oie']: raise NotImplementedError(task)
        last_sentence = None
        target: set[str] = set()
        for example in self.data:
            parts = normalize_quotes(example).split('\t')
            input_sentence, output_components = parts[0], parts[1:]
            if not input_sentence[0].isalnum():
                continue
            
            if len(output_components) < 2:
                continue
            segment = '(' + output_components[1].strip() + ' ; ' + output_components[0].strip()
            if len(output_components) >= 3:
                remaining = [s.strip() for s in output_components[2:]]
                if len(remaining) >= 1:
                    segment += ' ; ' + ' ; '.join(remaining)
            segment += ')'
            
            # # only include head, predicate, and first tail component
            # if len(output_components) < 3:
            #     continue
            # segment = '(%s ; %s ; %s)' % (output_components[1], output_components[0], output_components[2].rstrip())
            
            if input_sentence == last_sentence:
                target.add(segment)
            else:
                if last_sentence: yield last_sentence, ' '.join(target)
                last_sentence, target = input_sentence, set()
                

if __name__ == '__main__':
    o = CaRB(repo_path = sys.argv[1])
    o.to_json('oie', 'carb-oie')
    
