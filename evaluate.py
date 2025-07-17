## Rex's refactor of eval.py

from data import TrainData as TrainingData
from train import PRT5

if __name__ == '__main__':
    model: PRT5 = PRT5.load_from_checkpoint('outputs/checkpoints/srl-finetune.ckpt')
    dataset = TrainingData('outputs/datasets/conll-2012-srl.jsonl').loader(
        split = 'test',
        tokenizer = model.tokenizer,
        max_seq_length = 256,
        eval_batch_size = 32,
        num_workers = 4,
    )
    texts, outputs, targets = model.generate(dataset)
    for i in range(3):
        print(f"""
        =============== Generated Output #{i} ===============
        Text: {texts[0]},
        Output: {outputs[0]},
        Target: {targets[0]},
        """)