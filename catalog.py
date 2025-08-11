from utils import join_path


tasks = {
    'chunking': 'conll-2000-chunking.jsonl',
    'pr': 'wiki-20231101.en-pr.jsonl',
    'ner': 'conll-2003-ner.jsonl',
    'mlm': 'wiki-20231101.en-mlm.jsonl',
    'ner': 'conll-2003-ner.jsonl',
    'oie': 'oie-2016-oie.jsonl',
    'pos': 'conll-2003-pos.jsonl',
    're': 'conll-2004-re.jsonl',
    'srl': 'conll-2012-srl.jsonl',
}

def get_dataset_path(task: str, dataset_dir: str = 'outputs/datasets'):
    if task not in tasks:
        raise NotImplementedError(task)
    return join_path(dataset_dir, tasks[task])