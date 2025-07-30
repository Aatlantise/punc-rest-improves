import logging
import sys

from data.modules import PrepData


logging.basicConfig(
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level = logging.DEBUG,
    stream = sys.stdout
)
logger = logging.getLogger(__name__)


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
        last_sentence, target = None, None
        for example in self.data:
            parts = example.split('\t')
            sentence, rest = parts[0], parts[1:]
            if not sentence[0].isalnum(): continue
            rest[1], rest[0] = rest[0], rest[1] # swap verb and subject
            segment = '(' + '; '.join([w.strip() for w in rest]) + ') '
            if sentence == last_sentence:
                target += segment
            else:
                if last_sentence: yield last_sentence, target.rstrip()
                last_sentence, target = sentence, segment
                

if __name__ == '__main__':
    o = CaRB(repo_path = sys.argv[1])
    o.to_json('oie', 'carb')
    
