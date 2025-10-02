import sys

from data.modules import PrepData
from utils import logger
from utils.conllu import CoNLLU_File

logger = logger(__name__)


class Antilles(PrepData):
    
    def __init__(self, repo_path: str):
        """Loads dataset form hugging face"""
        super().__init__(hf_dataset = False)
        for split in ['train', 'dev', 'test']:
            f = CoNLLU_File(f'{repo_path}/ANTILLES/{split}.conllu')
            self.data += f.entries
        logger.debug('Data accumulated %d conllu entries', len(self.data))

    def src_tgt_pairs(self, task):
        if task not in ['pos']:
            raise NotImplementedError(f'Task {task} not implemented. ')
        for example in self.data:
            yield example.text, example.pos_str()


if __name__ == '__main__':
    o = Antilles(repo_path = sys.argv[1])
    o.to_json('pos', 'antilles-pos')