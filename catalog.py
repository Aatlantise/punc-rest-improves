# This file documents existing dataset locations
# So it's easier to use with `-d`; no need to type out full path
# For example `-d genia` when task is `ner` is the same as `-d outputs/datasets/genia-ner.jsonl'

from os.path import join

DATASET_DIR = 'outputs/datasets'

catalog = {
    'chunking': {
        'default': 'conll00',
        'ds': {
            'conll00': 'conll-2000-chunking.jsonl',
        },
    },
    'pr': {
        'default': 'wiki',
        'ds': {
            'wiki': 'wiki-20231101.en-pr.jsonl',
        },
    },
    'mlm': {
        'default': 'wiki',
        'ds': {
            'wiki': 'wiki-20231101.en-mlm.jsonl',
        },
    },
    'ner': {
        'default': 'conll03',
        'ds': {
            'conll03': 'conll-2003-ner.jsonl',
            'genia': 'genia-ner.jsonl',
            'multinerd': 'multinerd.fr-ner.jsonl',
            'ontonotes': 'ontonotes5-ner.jsonl',
            'partut': 'partut-ner.jsonl',
            'wikiann': 'wikiann.fr-ner.jsonl',
        },
    },
    'oie': {
        'default': 'oie',
        'ds': {
            'oie': 'oie-2016-oie.jsonl',
            'carb': 'carb-oie.jsonl',
        },
    },
    'pos': {
        'default': 'conll03',
        'ds': {
            'antilles': 'antilles-pos.jsonl',
            'conll00': 'conll-2000-pos.jsonl',
            'conll03': 'conll-2003-pos.jsonl',
        }
    },
    're': {
        'default': 'conll04',
        'ds': {
            'conll04': 'conll-2004-re.jsonl',
        },
    },
    'srl': {
        'default': 'conll12',
        'ds': {
            'conll12': 'conll-2012-srl.jsonl',
        }
    }
}

def get_dataset_path(
    task: str,
    ds_name: str = None,
):
    if task not in catalog.keys():
        raise NotImplementedError(task)
    task_obj = catalog[task]
    task_datasets = task_obj['ds']
    filename = task_datasets[ds_name or task_obj['default']]
    return join(DATASET_DIR, filename)