import json
import os

from argparse import ArgumentParser
from data.modules import TrainData, NumericTrainData
from importlib import import_module
from tasks.ner import score as object_generation_score
from train import PRT5, PRT5Numeric
from utils import logger, clean_split


PUNCTUATION_TO_REMOVE = {',', '.', '!', '?', '"', '�', '�', '�', "'"}
MAX_WORDS = 150
MAX_EXCERPTS = 450000


def normalize_text(text):
    """Lowercase and remove specific punctuation and capitalization."""
    text = text.lower()
    return ''.join(ch for ch in text if ch not in PUNCTUATION_TO_REMOVE)

def make_source_target(input):
    text = input['source']
    text = text[1:]
    text = text[:-1]
    text = text.replace(") (", " ")
    return {"source": normalize_text(text), "target" : text}

logger = logger()

def run(
    # task: str,
    # model_name: str,
    ckpt_path: str,
    data_path: str,
    max_seq_length: int = 512,
    eval_batch_size: int = 32,
    num_workers: int = 4,
):
    # if task not in ['srl', 'pos', 'oie', 'ner', 're', 'chunking', 'pr', 'sbd',
    #                 'glue_CoLA', 'glue_sst2', 'glue_mrpc', 'glue_stsb', 'glue_qqp',
    #                 'glue_mnli', 'glue_qnli', 'glue_rte', 'glue_wnli']:
    #     raise NotImplementedError(task)

    print(f"=============== PR evaluation on  {data_path.split('/')[-1].split('.')[0]} ===============")
    path = 'outputs/generated/verification_%s.jsonl' % data_path.split('/')[-1].split('.')[0]
    
    # data_paths = {
    #     'pr': 'outputs/datasets/wiki-20231101.en-pr.jsonl',
    #     'mlm': 'outputs/datasets/wiki-20231101.en-mlm.jsonl',
    #     'srl': 'outputs/datasets/conll-2012-srl.jsonl',
    #     'pos': 'outputs/datasets/conll-2003-pos.jsonl',
    #     'oie': 'outputs/datasets/oie-2016-oie.jsonl',
    #     'chunking': 'outputs/datasets/conll-2000-chunking.jsonl',
    #     're': 'outputs/datasets/conll-2004-re.jsonl',
    #     'ner': 'outputs/datasets/conll-2003-ner.jsonl',
    #     'glue_CoLA': 'outputs/datasets/cola_train.jsonl',
    #     'glue_sst2': 'outputs/datasets/sst2_train.jsonl',
    #     'glue_mrpc': 'outputs/datasets/mrpc_train.jsonl',
    #     'glue_stsb': 'outputs/datasets/stsb_train.jsonl',
    #     'glue_qqp': 'outputs/datasets/qqp_train.jsonl',
    #     'glue_mnli': 'outputs/datasets/mnli_train.jsonl',
    #     'glue_qnli': 'outputs/datasets/qnli_train.jsonl',
    #     'glue_rte': 'outputs/datasets/rte_train.jsonl',
    #     'glue_wnli': 'outputs/datasets/wnli_train.jsonl',
    # }
    
    texts, outputs, targets = [], [], []
    model = PRT5.load_from_checkpoint(ckpt_path)
    ds = TrainData(data_path)
    
    for split in ['train', 'dev', 'test']:
        ds.data[split] = [make_source_target(example) for example in ds.data[split]]


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
            if output != target:
                json.dump({'text': text, 'output': output, 'target': target}, f, ensure_ascii = False)
                f.write('\n')
    p, r, f1 = import_module('tasks.pr').score(texts, outputs, targets, strict = True)
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
    args = parser.parse_args()
    
    run(
        data_path = args.dataset_jsonl,
        ckpt_path = args.ckpt,
    )

# testPR.py -c outputs/checkpoints/T5PR1e4LRepoch40.ckpt -d outputs/datasets/wnli_train.jsonl
# testPR.py -c outputs/checkpoints/T5PR1e4LRepoch40.ckpt -d outputs/datasets/mnli_train.jsonl
# testPR.py -c outputs/checkpoints/T5PR1e4LRepoch40.ckpt -d outputs/datasets/mrpc_train.jsonl