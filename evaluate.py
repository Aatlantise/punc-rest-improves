## Rex's refactor of eval.py

from data import TrainData as TrainingData
from train import PRT5

if __name__ == '__main__':
    model: PRT5 = PRT5.load_from_checkpoint('outputs/checkpoints/pr-srl-final.ckpt')
    ds = TrainingData('outputs/datasets/conll-2012-srl.jsonl')
    dl = ds.loader(
        split = 'test',
        tokenizer = model.tokenizer,
        max_seq_length = 256,
        eval_batch_size = 32,
        num_workers = 4,
    )
    texts, outputs, targets = model.generate(dl)
    for i in range(3):
        print(f"""
        =============== Generated Output #{i} ===============
        Text: {texts[i]},
        Output: {outputs[i]},
        Target: {targets[i]},
        """)