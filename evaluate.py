## Rex's refactor of eval.py

import torch

from data.modules import TrainData as TrainingData
from train import PRT5

def prf1_score(
    texts: list[str],
    outputs: list[str],
    targets: list[str]
) -> tuple[float, float, float]:
    attempts = len(texts)
    if len(outputs) != attempts:
        raise ValueError(f'Output count: {len(outputs)}, not attempt count: {attempts}')
    if len(targets) != attempts:
        raise ValueError(f'Target count: {len(targets)}, not attempt count: {attempts}')
    gold, correct = 0, 0
    for text, output, target in zip(texts, outputs, targets):
        if output.strip() == target.strip():
            correct += 1
        gold += 1
    p = correct / attempts
    r = correct / gold
    f1 = 2 * p * r / (p + r)
    return p, r, f1

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
        print(f"""
        =============== Generated Output #{i} ===============
        Text: {texts[i]},
        Output: {outputs[i]},
        Target: {targets[i]},
        """)
    
