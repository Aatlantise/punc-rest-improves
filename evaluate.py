import logging
import sys
import torch

from data.modules import TrainData as TrainingData
from eval import pr_score
from train import PRT5

logging.basicConfig(
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level = logging.DEBUG,
    stream = sys.stdout
)
logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)

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
