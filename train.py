import lightning
import numpy as np
import random
import re
import torch

from argparse import ArgumentParser
from catalog import get_dataset_path
from data.modules import TrainData
from importlib import import_module
from lightning import Callback, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from models.t5 import PRT5
from typing import Callable
from utils import join_path, logger

logger = logger(__name__)
torch.autograd.set_detect_anomaly(True)


def set_seed(seed: int):
    """Seed all random components"""
    random.seed(seed)
    np.random.seed(seed)
    lightning.seed_everything(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
class MyCheckpoint(Callback):
    def __init__(
        self,
        epochs_to_save_at: list[int],
        save_dir: str = 'outputs/checkpoints',
        name: str = 'indiv-ckpt',
        validation_eval_metric: Callable[[list[str], list[str], list[str]], tuple[float, float, float]] = None,
    ):
        super().__init__()
        self.epochs_to_save_at = epochs_to_save_at or []
        self.save_dir = save_dir
        self.name = name
        self.eval_metric = validation_eval_metric

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.test(self.eval_metric)
        
        epoch = trainer.current_epoch
        if epoch in self.epochs_to_save_at:
            ckpt_path = join_path(self.save_dir, '%s%d.ckpt' % (self.name, epoch))
            trainer.save_checkpoint(ckpt_path)
            logger.info('Saved checkpoint at %s' % ckpt_path)

def run(
    ckpt_filename: str,
    data_path: str,
    epochs_to_save: list[int],
    save_last_epoch: bool,
    min_epochs: int,
    max_epochs: int,
    accelerator: str = 'gpu',
    adam_epsilon: float = 1e-8,
    devices: int = 1,
    eval_batch_size: int = 32,
    learning_rate: float = 3e-4,
    log_every_n_steps: int = 10,
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
    validation_eval_metric = None,
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
    
    trainer = Trainer(
        min_epochs = min_epochs,
        max_epochs = max_epochs,
        logger = TensorBoardLogger(save_dir = join_path(output_dir, 'logs'), name = ckpt_filename),
        callbacks = [
            ModelCheckpoint(
                dirpath = join_path(output_dir, 'checkpoints'),
                filename = '%s-{epoch}-{val_loss:.4f}' % ckpt_filename,
                save_top_k = save_top_k,
                verbose = True,
                monitor = monitor_metric,
                mode = 'min',
            ),
            LearningRateMonitor(logging_interval = 'step'),
            MyCheckpoint(
                epochs_to_save_at = epochs_to_save,
                name = ckpt_filename,
                validation_eval_metric = validation_eval_metric
            ),
        ],
        precision = precision,
        accelerator = accelerator,
        devices = devices,
        log_every_n_steps = log_every_n_steps,
        default_root_dir = output_dir,
        accumulate_grad_batches = 16,
    )
    logger.info('Initialized trainer')
    
    trainer.fit(model)
    logger.info('Fitted model')
    
    if save_last_epoch:
        final_ckpt_name = '%s%d.ckpt' % (ckpt_filename, max_epochs)
        final_save_path = join_path(output_dir, 'checkpoints', final_ckpt_name)
        trainer.save_checkpoint(final_save_path)
        logger.info(f'Saved last epoch: {final_save_path}')
    else:
        logger.info('Not saving last epoch. Continuing. ')
    
    try:
        trainer.test(model)
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
        '-n', '--ckpt-name',
        type = str, required = True,
        help = """
            Name the checkpoints that will be saved for this training.
            Timestamps, val_loss for each epoch might be appended.
            File extension will be appended.
            """
    )
    parser.add_argument(
        '-d', '--dataset',
        type = str,
        help = """
            A jsonl file containing training data,
            or a name of a dataset specified in the catalog.
            
            If left unprovided, a corresponding default jsonl will be used.
            """,
    )
    parser.add_argument(
        '-e', '--epochs',
        type = str, default = '3',
        help = """
            Number of epochs to run.
            
            For example, `3` means to run for exactly 3 epochs.
            `2-5` means to run for at least 2 epochs but at most 5 epochs.
            """
    )
    parser.add_argument(
        '--learning-rate',
        type = float, default = 3e-4,
        help = """
            Learning rate.
            """
    )
    parser.add_argument(
        '--max-seq-len',
        type = int, default = 512,
        help = """
            Max token length in a sequence.
            """
    )
    parser.add_argument(
        '--precision',
        type = str, default = 'bf16-mixed',
        help = """
            Precision for lightning trainer.
            
            Choose one of
            64, 64-true, 32, 32-true, 16, 16-mixed, bf16, bf16-mixed.
            """
    )
    parser.add_argument(
        '-r', '--resume-ckpt',
        type = str,
        help = """
            Checkpoint file to use at the beginning of the training.
            If left unprovided, t5-base will be used.
            """
    )
    parser.add_argument(
        '-s', '--save-epoch',
        action = 'append', type = int,
        help = """
            An epoch index (STARTS AT 0) to save. Can be provided multiple times.

            For example, `-s 3` will save the fourth epoch.
            `-s 2 -s 5` will save the third and sixth epochs.
            """
    )
    parser.add_argument(
        '--save-last-epoch',
        action = 'store_true',
        help = 'Save the last epoch of training as a checkpoint.'
    )
    parser.add_argument(
        '-k', '--save-top-k-epochs',
        type = int, default = '1',
        help = 'Save the top k checkpoints with lowest validation loss.'
    )
    parser.add_argument(
        '--seed',
        type = int, default = 42,
        help = 'Random seed for reproducibility. '
    )
    args = parser.parse_args()

    min_epochs, max_epochs = 0, 0
    if re.fullmatch(r'\d+-\d+', args.epochs):
        s = args.epochs.split('-')
        min_epochs, max_epochs = int(s[0]), int(s[1])
    elif re.fullmatch(r'\d+', args.epochs):
        min_epochs = int(args.epochs)
        max_epochs = min_epochs
    else:
        raise SyntaxError(f'Option -e/--epoch received invalid argument "{args.epochs}"')
    
    validation_eval_metric = None
    if args.task not in ['pr', 'mlm']:
        # only evaluate during validation for fine-tuning
        try:
            validation_eval_metric = import_module('tasks.' + args.task).score
        except:
            logger.warning(f'Eval metric for task {args.task} not found.')
            
    ds = args.dataset
    if not ds:
        ds = get_dataset_path(args.task)
    elif '.' not in ds:
        ds = get_dataset_path(args.task, ds_name = ds)
    
    logger.passthru(args.task, 'task')
    run(
        data_path = logger.passthru(ds, 'data path'),
        resume_ckpt = logger.passthru(args.resume_ckpt, 'resume checkpoint path'),
        ckpt_filename = logger.passthru(args.ckpt_name, 'checkpoint filename'),
        epochs_to_save = logger.passthru(args.save_epoch, 'epochs to save'),
        save_last_epoch = logger.passthru(args.save_last_epoch, 'save last epoch'),
        min_epochs = logger.passthru(min_epochs, 'min epochs'),
        max_epochs = logger.passthru(max_epochs, 'max epochs'),
        precision = logger.passthru(args.precision, 'precision'),
        save_top_k = logger.passthru(args.save_top_k_epochs, 'save top k'),
        seed = logger.passthru(args.seed, 'seed'),
        learning_rate = logger.passthru(args.learning_rate, 'learning rate'),
        validation_eval_metric = logger.passthru(validation_eval_metric, 'validation evaluation metric'),
    )