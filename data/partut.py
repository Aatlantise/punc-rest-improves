from data.modules import PrepData
from utils import logger

logger = logger(__name__)


class ParTUT(PrepData):
    
    def __init__(self):
        """Loads dataset form hugging face"""
        super().__init__('CATIE-AQ/universal_dependencies_fr_partut_fr_prompt_pos')
        
    @staticmethod
    def remove_prompt(text):
        """Remove the initial French from the text"""
        segments = text.split(':', 1)
        return segments[-1].lstrip()
    
    def src_tgt_pairs(self, task):
        if task not in ['pos']:
            raise NotImplementedError(f'Task {task} not implemented. ')
        last_source = ''
        for example in self:
            source = self.remove_prompt(example['inputs'])
            if source == last_source:
                continue # non-prompt part is the same
            target = example['targets']
            yield source, target
            last_source = source


if __name__ == '__main__':
    o = ParTUT()
    o.to_json('pos', 'partut-ner')