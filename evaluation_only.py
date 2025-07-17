import argparse
import logging
import os
import random
import json
import numpy as np
from tqdm import tqdm
from typing import List, Any, Union
from eval import pr_score, NER_eval
from main_fine_tune_conll import PRT5, get_punctuation_dataset, generate

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import torch


def run_NER(output_dir: str, eval_batch_size: int = 32, max_seq_length: int = 256):
    ckpt_path = os.path.join(output_dir, "checkpoints", "last.ckpt")
    model = PRT5.load_from_checkpoint(ckpt_path)
    texts, outputs, targets = generate(
        ckpt = None, # change to checkpoint if desired
        # ckpt = ckpt_path,
        model=model,
        input_dataset=get_punctuation_dataset(model.tokenizer, "test", max_seq_length),
        tokenizer=model.tokenizer,
        batch_size=eval_batch_size,
        # max_len=max_seq_length,
        shuffle=False
    )
    #  NER_eval(texts, outputs, targets, printer=logger.info)
    NER_eval(texts, outputs, targets)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()
    run_NER(output_dir=args.output_dir, eval_batch_size=32, max_seq_length=256)
