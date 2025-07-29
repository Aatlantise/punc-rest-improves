import logging
import sys

from data.modules import PrepData


logging.basicConfig(
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level = logging.DEBUG,
    stream = sys.stdout
)
logger = logging.getLogger(__name__)


class OIE2016(PrepData):

    def __init__(self, oie_path: str, split = 'train'):
        """Loads dataset from generated OpenIE corpus
        
        Refer to https://github.com/gabrielStanovsky/oie-benchmark
        """
        super().__init__(hf_dataset = False)
        with open(oie_path, 'r') as file:
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
    o = OIE2016(oie_path = './all.oie')
    o.to_json('oie', 'oie-2016')
    
