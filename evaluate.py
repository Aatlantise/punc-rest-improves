## Rex's refactor of eval.py

import logging
import sys
import torch

from data.modules import TrainData as TrainingData
from sklearn.metrics import precision_recall_fscore_support
from train import PRT5

logging.basicConfig(
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level = logging.DEBUG,
    stream = sys.stdout,
)
logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    model: PRT5 = PRT5.load_from_checkpoint('outputs/checkpoints/pr-srl.20250718-045803.epoch=1-val_loss=0.0413.ckpt')
    ds = TrainingData('outputs/datasets/conll-2012-srl.jsonl')
    dl = ds.loader(
        split = 'test',
        tokenizer = model.tokenizer,
        max_seq_length = 256,
        eval_batch_size = 32,
        num_workers = 4,
    )
    texts, outputs, targets = model.generate(dl)
    for i in range(10):
        print(
            f"""
            =============== Generated Output #{i} ===============
            Text: {texts[i]},
            Output: {outputs[i]},
            Target: {targets[i]},
            """
        )
    p, r, f1 = precision_recall_fscore_support(targets, outputs)
    print(
        f"""
        =============== Evaluation Result ===============
        Precision: {p},
        Recall: {r},
        F1: {f1},
        """
    )
    
