from utils import join_path


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
            'wiki': 'wiki-20231101-pr.jsonl',
        },
    },
    'mlm': {
        'default': 'wiki',
        'ds': {
            'wiki': 'wiki-20231101-mlm.jsonl',
        },
    },
    'ner': {
        'default': 'conll03',
        'ds': {
            'conll03': 'conll-2003-ner.jsonl',
            'ontonotes': 'ontonotes5-ner.jsonl',
            'genia': 'genia-ner.jsonl',
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

def get_dataset_path(task: str, dataset_dir: str = 'outputs/datasets'):
    if task not in catalog.keys():
        raise NotImplementedError(task)
    task_obj = catalog[task]
    task_datasets = task_obj['ds']
    default_file = task_datasets[task_obj['default']]
    return join_path(dataset_dir, default_file)