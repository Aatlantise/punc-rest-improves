## Rex's refactor of eval.py

import logging
import sys
import torch

from data.modules import TrainData as TrainingData
from eval import pr_score
from train import PRT5

logging.basicConfig(
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level = logging.DEBUG,
    stream = sys.stdout,
)
logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    print("=============== Baseline PR ===============")
    model: PRT5 = PRT5.load_from_checkpoint('outputs/checkpoints/pr.20250717-161054.epoch=1-val_loss=0.1053.ckpt')
    ds = TrainingData('outputs/datasets/conll-2012-srl-512t.jsonl')
    dl = ds.loader(
        split = 'test',
        tokenizer = model.tokenizer,
        max_seq_length = 512,
        eval_batch_size = 32,
        num_workers = 4,
    )
    texts, outputs, targets = model.generate(dl)
    for i in range(5):
        print(
            f"""
                =============== Generated Output #{i} ===============
                Text: {texts[i]},
                Output: {outputs[i]},
                Target: {targets[i]},
                """
        )
    f1 = pr_score(texts, outputs, targets)
    print(
        f"""
            =============== Evaluation Result ===============
            F1: {f1},
            """
    )
    
    print("=============== 256-Token SRL ===============")
    model: PRT5 = PRT5.load_from_checkpoint('outputs/checkpoints/pr-srl.20250718-045803.epoch=1-val_loss=0.0413.ckpt')
    ds = TrainingData('outputs/datasets/conll-2012-srl-256t.jsonl')
    dl = ds.loader(
        split = 'test',
        tokenizer = model.tokenizer,
        max_seq_length = 256,
        eval_batch_size = 32,
        num_workers = 4,
    )
    texts, outputs, targets = model.generate(dl)
    for i in range(5):
        print(
            f"""
            =============== Generated Output #{i} ===============
            Text: {texts[i]},
            Output: {outputs[i]},
            Target: {targets[i]},
            """
        )
    f1 = pr_score(texts, outputs, targets)
    print(
        f"""
        =============== Evaluation Result ===============
        F1: {f1},
        """
    )
    
    print("=============== 512-Token SRL ===============")
    model: PRT5 = PRT5.load_from_checkpoint('outputs/checkpoints/pr-srl-512tokens.20250721-095450.epoch=1-val_loss=0.0756.ckpt')
    ds = TrainingData('outputs/datasets/conll-2012-srl-512t.jsonl')
    dl = ds.loader(
        split = 'test',
        tokenizer = model.tokenizer,
        max_seq_length = 512,
        eval_batch_size = 32,
        num_workers = 4,
    )
    texts, outputs, targets = model.generate(dl)
    for i in range(5):
        print(
            f"""
            =============== Generated Output #{i} ===============
            Text: {texts[i]},
            Output: {outputs[i]},
            Target: {targets[i]},
            """
        )
    f1 = pr_score(texts, outputs, targets)
    print(
        f"""
        =============== Evaluation Result ===============
        F1: {f1},
        """
    )
    
