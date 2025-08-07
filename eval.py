import json
import os

from argparse import ArgumentParser
from data.modules import TrainData
from importlib import import_module
from tasks.ner import score as object_generation_score
from train import PRT5
from utils import logger, clean_split

logger = logger()


def multitask_score(texts, outputs, targets, printer = print):
    # assume NER-OIE multitask model
    ner_outputs = []
    ner_targets = []
    oie_outputs = []
    oie_targets = []

    for output, target in zip(outputs, targets):
        ner_os = []
        oie_os = []
        ner_ts = []
        oie_ts = []
        output = output.replace('() ', '')
        target = target.replace('() ', '')

        for o in clean_split(output):
            if any([k in o for k in ['LOC: ', 'PER: ', 'ORG: ', 'PRO: ']]):
                ner_os.append(f"({o})")
            else:
                oie_os.append(f"({o})")
        for t in clean_split(target):
            if any([k in t for k in ['LOC: ', 'PER: ', 'ORG: ', 'PRO: ']]):
                ner_ts.append(f"({t})")
            else:
                oie_ts.append(f"({t})")
        ner_outputs.append(' '.join(ner_os))
        ner_targets.append(' '.join(ner_ts))
        oie_outputs.append(' '.join(oie_os))
        oie_targets.append(' '.join(oie_ts))

    object_generation_score(texts, ner_outputs, ner_targets)
    object_generation_score(texts, oie_outputs, oie_targets)


def run(
    task: str,
    model_name: str,
    ckpt_path: str,
    data_path: str,
    max_seq_length: int = 512,
    eval_batch_size: int = 32,
    num_workers: int = 4,
    strict: bool = True,
):
    if task not in ['srl', 'pos', 'oie', 'ner', 're', 'chunking', 'pr']:
        raise NotImplementedError(task)
    
    print(f"=============== Model {model_name} {task} Evaluation ===============")
    path = 'outputs/generated/%s.jsonl' % model_name.split(' ', 1)[0]
    
    default_data_paths = {
        'pr': 'outputs/datasets/wiki-20231101.en-pr.jsonl',
        'mlm': 'outputs/datasets/wiki-20231101.en-mlm.jsonl',
        'srl': 'outputs/datasets/conll-2012-srl.jsonl',
        'pos': 'outputs/datasets/conll-2003-pos.jsonl',
        'oie': 'outputs/datasets/oie-2016-oie.jsonl',
        'chunking': 'outputs/datasets/conll-2000-chunking.jsonl',
        're': 'outputs/datasets/conll-2004-re.jsonl',
        'ner': 'outputs/datasets/conll-2003-ner.jsonl',
    }
    
    texts, outputs, targets = [], [], []
    if os.path.isfile(path):
        logger.info('Restoring outputs from %s.' % path)
        with open(path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj['text'])
                outputs.append(obj['output'])
                targets.append(obj['target'])
    else:
        logger.info(f'Loading model {model_name} from checkpoint {ckpt_path}')
        model = PRT5.load_from_checkpoint(ckpt_path)
        
        logger.info(f'Loading dataset from path {data_path}')
        data_path = data_path or default_data_paths[task]
        ds = TrainData(data_path)
        
        logger.info('Initializing dataloader. ')
        dl = ds.loader(
            split = 'test',
            tokenizer = model.tokenizer,
            max_seq_length = max_seq_length,
            eval_batch_size = eval_batch_size,
            num_workers = num_workers,
        )
        
        logger.info('Generating outputs.')
        texts, outputs, targets = model.generate(dl)
        
        logger.info('Backing up outputs to %s.' % path)
        with open(path, 'w') as f:
            for i in range(len(texts)):
                text = texts[i]
                output = outputs[i] if i < len(outputs) else None
                target = targets[i] if i < len(targets) else None
                json.dump({'text': text, 'output': output, 'target': target}, f, ensure_ascii = False)
                f.write('\n')
    
    logger.info(f'Evaluating {task} score.')
    p, r, f1 = import_module('tasks.' + task).score(texts, outputs, targets, strict = strict)
    print(
        f"""
        =============== Evaluation Result ===============
        Precision: {p},
        Recall: {r},
        F1: {f1},
        """
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'task',
        type = str,
        help = 'The evaluation to perform.',
    )
    parser.add_argument(
        '-c', '--ckpt',
        type = str, required = True,
        help = 'Path to the checkpoint to be evaluated. '
    )
    parser.add_argument(
        '-d', '--dataset-jsonl',
        type = str,
        help = """
            A jsonl file containing evaluating data.
            If left unprovided, a corresponding default jsonl will be used.
            """,
    )
    parser.add_argument(
        '-n', '--model-name',
        type = str, required = True,
        help = 'Name the model that will be evaluated, to be used in result printing. '
    )
    parser.add_argument(
        '--strict',
        action = 'store_true',
        help = 'Use a stricter metric for evaluation. See implementation detail for each task.'
    )
    args = parser.parse_args()
    
    run(
        task = args.task,
        model_name = args.model_name,
        ckpt_path = args.ckpt,
        data_path = args.dataset_jsonl,
        strict = args.strict,
    )
