from data.modules import PrepData
from utils import logger

logger = logger()


class OIE2016(PrepData):

    def __init__(self, repo_path: str):
        """Loads dataset from generated OpenIE corpus
        
        Refer to https://github.com/gabrielStanovsky/oie-benchmark.
        Need to run a script from the repo to convert QA-SRL data into OpenIE
        """
        super().__init__(hf_dataset = False)
        with open(f'{repo_path}/oie_corpus/all.oie', 'r') as file:
            for line in file:
                self.data.append(line)

    def src_tgt_pairs(self, task: str):
        if task not in ['oie']: raise NotImplementedError(task)
        last_sentence, target = None, None
        for example in self.data:
            parts = example.split('\t')
            sentence, rest = parts[0], parts[2:] # parts[1] is the pure verb
            if not sentence[0].isalnum(): continue
            rest[1], rest[0] = rest[0], rest[1] # swap verb and subject
            segment = '(' + '; '.join([w.strip() for w in rest]) + ') '
            if sentence == last_sentence:
                target += segment
            else:
                if last_sentence: yield last_sentence, target.rstrip()
                last_sentence, target = sentence, segment
                

if __name__ == '__main__':
    o = OIE2016(repo_path = sys.argv[1])
    o.to_json('oie', 'oie-2016-oie')
    
