import lightning
import numpy as np
import random
import re
import torch

from functools import partial
from argparse import ArgumentParser
from data.modules import TrainData, NumericTrainData, IntTrainData
from datetime import datetime
from lightning import Callback, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from models.t5 import PRT5, PRT5Numeric
from models.felflarebert import felflarebert
from os.path import join as join_paths
from utils import logger

logger = logger()
torch.autograd.set_detect_anomaly(True)


def set_seed(seed: int):
    """Seed all random components"""
    random.seed(seed)
    np.random.seed(seed)
    lightning.seed_everything(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
class IndividualCheckpoints(Callback):
    def __init__(self, epochs_to_save_at: list[int], save_dir: str = 'outputs/checkpoints', name: str = 'indiv-ckpt'):
        super().__init__()
        self.epochs_to_save_at = epochs_to_save_at or []
        self.save_dir = save_dir
        self.name = name

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch in self.epochs_to_save_at:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            ckpt_path = join_paths(self.save_dir, '%s-%s-epoch%s.ckpt' % (self.name, timestamp, epoch))
            trainer.save_checkpoint(ckpt_path)
            logger.info('Saved checkpoint at %s' % ckpt_path)

def apply_prefix(task, x):
    # return f"{task}: {x['source']}", x['target']
    return {"source": f"{task}: {x['source']}", "target" : x['target']}

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
):
    """Run training on data path"""
    torch.set_float32_matmul_precision('medium')
    set_seed(seed)
    
    glue_datasets = ['outputs/datasets/cola_train.jsonl',
                    'outputs/datasets/sst2_train.jsonl',
                    'outputs/datasets/mrpc_train.jsonl',
                    'outputs/datasets/stsb_train.jsonl',
                    'outputs/datasets/qqp_train.jsonl',
                    # 'outputs/datasets/mnli_train.jsonl', # this dataset's has two test sets matched and mismatched so ignore for now
                    'outputs/datasets/mnli_matched_train.jsonl', # these are just copies of mnli_train
                    'outputs/datasets/mnli_mismatched_train.jsonl',
                    'outputs/datasets/qnli_train.jsonl',
                    'outputs/datasets/rte_train.jsonl',
                    'outputs/datasets/wnli_train.jsonl']

    if "bert" in model_name_or_path.lower():
        logger.debug("Using bert model, loading IntTrainData")
        if data_path in glue_datsets:
          dev_path = data_path.replace("_train.jsonl", "_dev.jsonl")
          training_data = IntTrainData(data_path, dev_path)
          print("glue dataset loaded")
        else:
          training_data = IntTrainData(data_path)
        # training_data = IntTrainData(data_path)
        model = felflarebert.load_from_checkpoint(resume_ckpt) if resume_ckpt else felflarebert(
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
            num_labels = training_data.num_labels,
        )
        logger.info("felflare bert model being used (overwrites previous model)")
    else:
        if data_path in glue_datasets:
            dev_path = data_path.replace("_train.jsonl", "_dev.jsonl")
            training_data = TrainData(data_path, dev_path)
            # print("glue dataset loaded")
        else:
            training_data = TrainData(data_path)
        
        if (data_path == 'outputs/datasets/stsb_train.jsonl'):
            logger.debug("stsb regression")
            dev_path = data_path.replace("_train.jsonl", "_dev.jsonl")
            training_data = NumericTrainData(data_path, dev_path)
            print("glue dataset loaded")
        logger.info(f'Loaded training data from {data_path}')

        if (data_path in glue_datasets):
            # apply task prefix:
            logger.debug(f"Applying task prefix {data_path.split('/')[-1]}")
            for split in ['train', 'dev', 'test']:
                training_data.data[split] = [apply_prefix(data_path.split('/')[-1].split('_')[0], x) for x in training_data.data[split]]
        
        
        if (data_path == 'outputs/datasets/stsb_train.jsonl'):
            logger.debug("stsb regression")
            model = PRT5Numeric(
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
            if resume_ckpt:
                ckpt = torch.load(resume_ckpt, map_location='cpu')
                PR_model = PRT5.load_from_checkpoint(resume_ckpt)
                model.model.load_state_dict(PR_model.state_dict(), strict=False)
            # Did not have regression head for PR training
            # model = PRT5Numeric.load_from_checkpoint(resume_ckpt) if resume_ckpt else PRT5Numeric(
            #     adam_epsilon = adam_epsilon,
            #     eval_batch_size = eval_batch_size,
            #     learning_rate = learning_rate,
            #     max_seq_length = max_seq_length,
            #     model = model_name_or_path,
            #     num_train_epochs = num_train_epochs,
            #     num_workers = num_workers,
            #     train_batch_size = train_batch_size,
            #     warmup_steps = warmup_steps,
            #     weight_decay = weight_decay,
            # )
        else:
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
        min_epochs = min_epochs,
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
            LearningRateMonitor(logging_interval = 'step'),
            IndividualCheckpoints(epochs_to_save_at = epochs_to_save, name = ckpt_filename), # Save 1st and 10th epochs
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
            tokenizer = model.tokenizer,
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
        '-m', '--model-name-or-path',
        type = str, default = 'google-t5/t5-base',
        help = 'The model name or path to use.',
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
        '-d', '--dataset-jsonl',
        type = str,
        help = """
            A jsonl file containing training data.
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
        '-s', '--epoch-to-save',
        action = 'append', type = int,
        help = """
            An epoch index (STARTS AT 0) to save. Can be provided multiple times.

            For example, `-s 3` will save the fourth epoch.
            `-s 2 -s 5` will save the third and sixth epochs.
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
        action = 'store_true',
        help = 'Save the last epoch of training as a checkpoint.'
    )
    parser.add_argument(
        '-k', '--save-top-k',
        type = int, default = '1',
        help = 'Save the top k checkpoints with lowest validation loss.'
    )
    args = parser.parse_args()
    
    default_data_paths = {
        'pr': 'outputs/datasets/wiki-20231101.en-pr.jsonl',
        'mlm': 'outputs/datasets/wiki-20231101.en-mlm.jsonl',
        'srl': 'outputs/datasets/conll-2012-srl.jsonl',
        'pos': 'outputs/datasets/conll-2003-pos.jsonl',
        'oie': 'outputs/datasets/oie-2016-oie.jsonl',
        'chunking': 'outputs/datasets/conll-2000-chunking.jsonl',
        're': 'outputs/datasets/conll-2004-re.jsonl',
        'ner': 'outputs/datasets/conll-2003-ner.jsonl',
        'glue_CoLA': 'outputs/datasets/cola_train.jsonl',
        'glue_sst2': 'outputs/datasets/sst2_train.jsonl',
        'glue_mrpc': 'outputs/datasets/mrpc_train.jsonl',
        'glue_stsb': 'outputs/datasets/stsb_train.jsonl',
        'glue_qqp': 'outputs/datasets/qqp_train.jsonl',
        'glue_mnli_matched': 'outputs/datasets/mnli_matched_train.jsonl',
        'glue_mnli_mismatched': 'outputs/datasets/mnli_mismatched_train.jsonl',
        'glue_qnli': 'outputs/datasets/qnli_train.jsonl',
        'glue_rte': 'outputs/datasets/rte_train.jsonl',
        'glue_wnli': 'outputs/datasets/wnli_train.jsonl',
    }
    # default_pr_ckpt = 'outputs/checkpoints/pr.20250717-161054.epoch=1-val_loss=0.1053.ckpt'
    default_pr_ckpt = 'outputs/checkpoints/T5PR1e4LR.20250823-112452.epoch=1-val_loss=0.1037.ckpt'
    
    if not args.dataset_jsonl and args.task not in default_data_paths.keys():
        raise NotImplementedError(args.task)
    
    min_epochs, max_epochs = 0, 0
    if re.fullmatch(r'\d+-\d+', args.epochs):
        s = args.epochs.split('-')
        min_epochs, max_epochs = int(s[0]), int(s[1])
    elif re.fullmatch(r'\d+', args.epochs):
        min_epochs = int(args.epochs)
        max_epochs = min_epochs
    else:
        raise SyntaxError(f'Option -e/--epoch received invalid argument "{args.epochs}"')
    
    logger.debug(f"Data path is {args.dataset_jsonl or default_data_paths[args.task]}")
    logger.debug(f"Resuming checkpoint from {args.resume_ckpt or default_pr_ckpt if args.task not in ['pr', 'mlm'] else None}")
    logger.debug(f"Checkpoints will be named with prefix {args.ckpt_name}")
    logger.debug(f"Saving epochs {args.epoch_to_save}")
    logger.debug(f"Saving top {args.save_top_k} epochs")
    logger.debug(f"Whether to save last epoch: {args.save_last_epoch}")
    logger.debug(f"Epochs to run: min {min_epochs}, max {max_epochs}")
    print("model_name_or_path", args.model_name_or_path, "and resume_ckpt", args.resume_ckpt, "\n\n")
    
    run(
        data_path = args.dataset_jsonl or default_data_paths[args.task],
        resume_ckpt = args.resume_ckpt,
        ckpt_filename = args.ckpt_name,
        epochs_to_save = args.epoch_to_save,
        save_last_epoch = args.save_last_epoch,
        min_epochs = min_epochs,
        max_epochs = max_epochs,
        save_top_k = args.save_top_k,
        model_name_or_path = args.model_name_or_path,
    )