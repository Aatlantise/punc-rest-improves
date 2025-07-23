import json
import logging
import os
import sys
import torch

from data.modules import TrainData as TrainingData
from data.conll_2012 import CoNLL2012
from train import PRT5

logging.basicConfig(
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level = logging.DEBUG,
    stream = sys.stdout
)
logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)

def multiset_intersection(a, b):
    """The key types should be hashable of course, and the values numerics"""
    total = 0
    for key in a.keys() & b.keys():
        total += min(a[key], b[key])
    return total

def srl_eval(texts: list[str], outputs: list[str], targets: list[str]) -> tuple[float, float, float]:
    """Calculate precision, recall, and F1 score"""
    total = min(len(texts), len(outputs), len(targets))
    logger.info('SRL eval: length %d' % total)
    if total != len(texts):
        logger.warning('SRL eval: length mismatch: there are %d texts instead of %d' % (len(texts), total))
    if total != len(outputs):
        logger.warning('SRL eval: length mismatch: there are %d outputs instead of %d' % (len(outputs), total))
    if total != len(targets):
        logger.warning('SRL eval: length mismatch: there are %d targets instead of %d' % (len(targets), total))
    
    relevant_retrieved_instances, retrieved_instances, relevant_instances = 0, 0, 0
    for i in range(total):
        text, output, target = texts[i], outputs[i], targets[i]
        output_dict, output_label_count = CoNLL2012.unserialize(output)
        retrieved_instances += output_label_count
        target_dict, target_label_count = CoNLL2012.unserialize(target)
        relevant_instances += target_label_count
        if output_dict == target_dict:
            relevant_retrieved_instances += target_label_count
            continue
        output_verbs = [a for (a, b) in output_dict]
        target_verbs = [a for (a, b) in target_dict]
        if output_verbs == target_verbs:
            for j in range(len(output_verbs)):
                _, output_verb_frame = output_dict[j]
                _, target_verb_frame = target_dict[j]
                for label in output_verb_frame.keys() & target_verb_frame.keys():
                    relevant_retrieved_instances += multiset_intersection(
                        output_verb_frame[label],
                        target_verb_frame[label],
                    )
    
    precision = relevant_retrieved_instances / retrieved_instances
    recall = relevant_retrieved_instances / relevant_instances
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
        

def run(
    model_name: str,
    ckpt_path: str,
    data_path: str,
    max_seq_length: int,
    eval_batch_size: int = 32,
    num_workers: int = 4,
):
    print(f"=============== {model_name} ===============")
    model = PRT5.load_from_checkpoint(ckpt_path)
    logger.info(f'Loaded model {model_name} from checkpoint {ckpt_path}')
    ds = TrainingData(data_path)
    logger.info(f'Loaded dataset from path {data_path}')
    dl = ds.loader(
        split = 'test',
        tokenizer = model.tokenizer,
        max_seq_length = max_seq_length,
        eval_batch_size = eval_batch_size,
        num_workers = num_workers,
    )
    logger.info(f'Initialized dataloader. Generating outputs...')
    path = 'outputs/generated/%s.jsonl' % model_name.split(' ', 1)[0]
    texts, outputs, targets = [], [], []
    
    # Back up / restore generated outputs
    if os.path.isfile(path):
        with open(path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj['text'])
                outputs.append(obj['output'])
                targets.append(obj['target'])
    else:
        texts, outputs, targets = model.generate(dl)
        for text, output, target in zip(texts, outputs, targets):
            with open(path, 'w') as f:
                json.dump({'text': text, 'output': output, 'target': target}, f, ensure_ascii = False)
                f.write('\n')
                
    for i in range(5):
        print(
            f"""
            =============== Generated Output #{i} ===============
            Text: {texts[i]},
            Output: {outputs[i]},
            Target: {targets[i]},
            """
        )
    p, r, f1 = srl_eval(texts, outputs, targets)
    print(
        f"""
        =============== Evaluation Result ===============
        Precision: {p},
        Recall: {r},
        F1: {f1},
        """
    )

if __name__ == '__main__':
    run(
        model_name = 'Baseline-T5 on SRL',
        ckpt_path = 'outputs/checkpoints/srl-512tokens.20250722-111511.epoch=1-val_loss=0.0759.ckpt',
        data_path = 'outputs/datasets/conll-2012-srl-512t.jsonl',
        max_seq_length = 512,
    )
    run(
        model_name = 'PR-T5 on SRL',
        ckpt_path = 'outputs/checkpoints/pr-srl-512tokens.20250721-095450.epoch=1-val_loss=0.0756.ckpt',
        data_path = 'outputs/datasets/conll-2012-srl-512t.jsonl',
        max_seq_length = 512,
    )
