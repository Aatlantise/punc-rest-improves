import sys

from data.modules import PrepData
from tasks.oie import normalize_quotes
from utils import logger

logger = logger(__name__)


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
        if task not in ['oie']:
            raise NotImplementedError(task)
        last_sentence = None
        target: set[str] = set()
        for example in self:
            parts = normalize_quotes(example).split('\t')
            input_sentence, output_components = parts[0], parts[2:] # parts[1] is the simple verb
            if not input_sentence[0].isalnum():
                continue
                
            if len(output_components) < 2:
                continue
            segment = '(' + output_components[1].strip() + ' ; ' + output_components[0].strip()
            if len(output_components) >= 3:
                segment += ' ; ' + ' ; '.join([s.strip() for s in output_components[2:]])
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
    o = OIE2016(repo_path = sys.argv[1])
    o.to_json('oie', 'oie-2016-oie')
    
