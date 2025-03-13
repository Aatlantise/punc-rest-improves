import os
import nltk
import argparse
from main import T5GenericFineTuner, run, get_generic_dataset, generate, object_generation_score

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

f = open("openie4.txt", "w")
printer = f.write


for epoch, save_dir, checkpoint, model, seed in [
    (10, "lightning_logs/base-openie4-10epochs/checkpoints/0/", None, "t5-base", 0),
    (10, "lightning_logs/base-openie4-10epochs/checkpoints/1/", None, "t5-base", 1),
    (10, "lightning_logs/base-openie4-10epochs/checkpoints/2/", None, "t5-base", 2),
    (10, "lightning_logs/base-openie4-10epochs/checkpoints/3/", None, "t5-base", 3),
    (10, "lightning_logs/base-openie4-10epochs/checkpoints/4/", None, "t5-base", 4),
    (50, "lightning_logs/base-mpm-openie4-50epochs/checkpoints/0/", "/data/t5-mpm-checkpoints/base-mpm-40epochs.ckpt",
     "t5-base", 0),
    (50, "lightning_logs/base-mpm-openie4-50epochs/checkpoints/1/", "/data/t5-mpm-checkpoints/base-mpm-40epochs.ckpt",
     "t5-base", 1),
    (50, "lightning_logs/base-mpm-openie4-50epochs/checkpoints/2/", "/data/t5-mpm-checkpoints/base-mpm-40epochs.ckpt",
     "t5-base", 2),
    (50, "lightning_logs/base-mpm-openie4-50epochs/checkpoints/3/", "/data/t5-mpm-checkpoints/base-mpm-40epochs.ckpt",
     "t5-base", 3),
    (50, "lightning_logs/base-mpm-openie4-50epochs/checkpoints/4/", "/data/t5-mpm-checkpoints/base-mpm-40epochs.ckpt",
     "t5-base", 4),
]:

    epoch = epoch
    model_name = model
    nltk.download('punkt')
    if 'small' in model_name:
        batch_size = 64
    elif 'base' in model_name:
        batch_size = 32
    elif 'large' in model_name:
        batch_size = 16
    else:
        print(model_name)
        raise ValueError("This model name is no currently supported.")
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
        fp_16=True, # if you want to enable 16-bit training then install apex and set this to true
        opt_level='O1', # you can find out more on optimisation levels here
        # https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=0.5, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
    )

    args = argparse.Namespace(**args_dict)
    model = T5GenericFineTuner(args, "oie", "openie4")
    tokenizer = model.tokenizer

    a = os.path.isdir(args.output_dir)
    b = 'last.ckpt' in os.listdir(args.output_dir) if a else False
    monitoring_value = 'val_f1' if model.epoch_end_eval else 'val_loss'
    if b:
        print(f"{args.output_dir}last.ckpt exists. Evaluating...")
        pass
    else:
        print(f"{args.output_dir}last.ckpt does not exist. Training...")
        model = run(args, model, resume_from_checkpoint=checkpoint, monitor=monitoring_value, seed=seed)

    ckpt = args.output_dir + 'last.ckpt'

    printer(f"Task: {model.task}, dataset: {model.dataset_file}, seed: {seed}\n")
    test_dataset = get_generic_dataset(model.tokenizer, model.dataset_file, 'test', model.prompt, model.task)
    texts, outputs, targets = generate(ckpt, model, test_dataset, model.tokenizer, args.eval_batch_size, skip_special_tokens=True)
    object_generation_score(texts, outputs, targets, printer=printer)
