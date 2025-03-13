import os
import nltk
import logging
import pickle
import argparse
import logging
import sys
from io import StringIO

from datasets import load_dataset
from transformers import AutoTokenizer, T5Tokenizer
from main import GenericDataset, T5GenericFineTuner, run, get_generic_dataset, generate, object_generation_score, \
    text2carb, get_carb_dataset, multitask_score, mpm_score, sbd_score, sequence_tagging_score

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

f = open("pr.txt", "w")
printer = f.write


for epoch, save_dir, checkpoint, model in [
    (40, "lightning_logs/gpt2-medium-mpm/checkpoints/", None, "gpt2-medium")
]:

    epoch = epoch
    model_name = model
    nltk.download('punkt')
    batch_size = 64
    NUM_EPOCHS = epoch
    MAX_LEN = 256

    args_dict = dict(
        data_dir="", # path for data files
        output_dir=save_dir, # path to save the checkpoints
        model_name_or_path=model_name,
        tokenizer_name_or_path=model_name,
        max_seq_length=MAX_LEN,
        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_train_epochs=NUM_EPOCHS,
        gradient_accumulation_steps=16,
        n_gpu=1,
        early_stop_callback=True,
        fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
        opt_level='O1', # you can find out more on optimisation levels here
        # https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=0.5, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
    )

    args = argparse.Namespace(**args_dict)
    model = T5GenericFineTuner(args, "mpm", "")
    tokenizer = model.tokenizer

    a = os.path.isdir(args.output_dir)
    b = 'last.ckpt' in os.listdir(args.output_dir) if a else False
    monitoring_value = 'val_f1' if model.epoch_end_eval else 'val_loss'
    if b:
        print(f"{args.output_dir}last.ckpt exists. Evaluating...")
        pass
    else:
        print(f"{args.output_dir}last.ckpt does not exist. Training...")
        model = run(args, model, resume_from_checkpoint=checkpoint, monitor=monitoring_value, seed='random')

    ckpt = args.output_dir + 'last.ckpt'

    printer(f"Task: {model.task}, dataset: {model.dataset_file}")
    test_dataset = get_generic_dataset(model.tokenizer, model.dataset_file, 'test', model.prompt, model.task)
    texts, outputs, targets = generate(ckpt, model, test_dataset, model.tokenizer, args.eval_batch_size, skip_special_tokens=True)
    mpm_score(texts, outputs, targets, printer=printer)
