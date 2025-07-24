## Rex's refactor of main.py

import lightning
import logging
import numpy as np
import random
import sys
import torch

from argparse import ArgumentParser
from data.modules import TrainData
from datetime import datetime
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from models.t5 import PRT5
from os.path import join as join_paths

logging.basicConfig(
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level = logging.DEBUG,
    stream = sys.stdout,
)
logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)


def set_seed(seed: int):
    """Seed all random components"""
    random.seed(seed)
    np.random.seed(seed)
    lightning.seed_everything(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run(
    ckpt_filename: str,
    data_path: str,
    save_last_epoch: bool,
    accelerator: str = 'gpu',
    adam_epsilon: float = 1e-8,
    devices: int = 1,
    eval_batch_size: int = 32,
    learning_rate: float = 3e-4,
    log_every_n_steps: int = 10,
    max_epochs = 3,
    max_seq_length: int = 512,
    model_name_or_path = 'google-t5/t5-base',
    monitor_metric: str = 'val_loss',
    num_train_epochs: int = 3,
    num_workers: int = 4,
    output_dir = 'outputs',
    precision: str = 'bf16-mixed',
    resume_ckpt: str = None,
    save_top_k: int = 1,
    seed: int = 42,
    train_batch_size: int = 32,
    warmup_steps: int = 0,
    weight_decay: float = 0.01,
):
    """Run training on data path"""
    torch.set_float32_matmul_precision('medium')
    set_seed(seed)
    
    training_data = TrainData(data_path)
    logger.info(f'Loaded training data from {data_path}')
    
    model = PRT5.load_from_checkpoint(resume_ckpt) if resume_ckpt else PRT5(
        adam_epsilon = adam_epsilon,
        eval_batch_size = eval_batch_size,
        learning_rate = learning_rate,
        max_seq_length = max_seq_length,
        model = model_name_or_path,
        num_train_epochs = num_train_epochs,
        num_workers = num_workers,
        train_batch_size = train_batch_size,
        warmup_steps = warmup_steps,
        weight_decay = weight_decay,
    )
    model.store_data(training_data)
    logger.info('Initialized model')
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    trainer = Trainer(
        max_epochs = max_epochs,
        logger = TensorBoardLogger(save_dir = join_paths(output_dir, 'logs'), name = ckpt_filename),
        callbacks = [
            ModelCheckpoint(
                dirpath = join_paths(output_dir, 'checkpoints'),
                filename = '%s.%s.{epoch}-{val_loss:.4f}' % (ckpt_filename, timestamp),
                save_top_k = save_top_k,
                verbose = True,
                monitor = monitor_metric,
                mode = 'min',
            ),
            LearningRateMonitor(logging_interval = 'step')
        ],
        precision = precision,
        accelerator = accelerator,
        devices = devices,
        log_every_n_steps = log_every_n_steps,
        default_root_dir = output_dir,
    )
    logger.info('Initialized trainer')
    
    trainer.fit(model)
    logger.info('Fitted model')
    
    if save_last_epoch:
        final_ckpt_name = '%s-final.ckpt' % ckpt_filename
        trainer.save_checkpoint(join_paths(output_dir, 'checkpoints', final_ckpt_name))
        logger.info(f'Saved last epoch to {final_ckpt_name}')
    else:
        logger.info('Not saving last epoch. Continuing. ')
    
    try:
        trainer.test(model, dataloaders = training_data.loader(
            split = 'test',
            tokenizer = model.tokenizer(),
            max_seq_length = max_seq_length,
            eval_batch_size = eval_batch_size,
            num_workers = num_workers,
        ))
        logger.info('Tested model')
    except Exception as e:
        logger.warning('Trainer test did not run. Error below:')
        logger.warning(e)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'task',
        type = str,
        help = 'The training task to perform.',
    )
    parser.add_argument(
        '-d', '--dataset-jsonl',
        type = str,
        help = """
            A jsonl file containing training data.
            If left unprovided, a corresponding default jsonl will be used.
            """,
    )
    parser.add_argument(
        '-n', '--ckpt-name',
        type = str,
        help = """
            Name the checkpoints that will be saved for this training.
            Timestamps, val_loss for each epoch might be appended.
            File extension will be appended.
            """
    )
    parser.add_argument(
        '-r', '--resume-ckpt',
        type = str,
        help = """
            Checkpoint file to use at the beginning of the training.
            If left unprovided for a pre-training task, t5-base will be used.
            If left unprovided for a fine-tuning task, a checkpoint pre-trained on PR will be used.
            """
    )
    parser.add_argument(
        '--save-last-epoch',
        action = 'store_false',
        help = 'Save the last epoch of training as a checkpoint.'
    )
    args = parser.parse_args()
    
    match args.task:
        case 'pr':
            run(
                data_path = 'outputs/datasets/wiki-20231101.en-pr.jsonl',
                resume_ckpt = args.resume_ckpt,
                ckpt_filename = args.ckpt_name or 'pr',
                save_last_epoch = args.save_last_epoch,
            )
        case 'srl':
            run(
                data_path = 'outputs/datasets/conll-2012-srl-512t.jsonl',
                resume_ckpt = args.resume_ckpt or 'outputs/checkpoints/pr.20250717-161054.epoch=1-val_loss=0.1053.ckpt',
                ckpt_filename = args.ckpt_name or 'pr-srl-512t',
                save_last_epoch = args.save_last_epoch,
            )
        case _:
            raise Exception('Task %s has not been implemented. Aborting...' % args.task)