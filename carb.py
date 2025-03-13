from main import get_carb_dataset, text2carb
import argparse
from main import T5GenericFineTuner

args_dict = dict(
    data_dir="",  # path for data files
    output_dir="",  # path to save the checkpoints
    model_name_or_path="t5-base",
    tokenizer_name_or_path="t5-base",
    max_seq_length=256,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=32,
    eval_batch_size=32,
    num_train_epochs=50,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=True,
    fp_16=True,  # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1',  # you can find out more on optimisation levels here
    # https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=0.5,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)

args = argparse.Namespace(**args_dict)


def get_carb_output(task_name, seed):
    model = T5GenericFineTuner(args, "oie", task_name)
    carb = get_carb_dataset(model.tokenizer)
    checkpoint = f"/data/checkpoints/base-{task_name}-10epochs/checkpoints/{str(seed)}/last.ckpt"
    text2carb(checkpoint, model, carb, model.tokenizer, args.max_seq_length, args.train_batch_size, seed=seed)
    checkpoint = f"/data/checkpoints/base-mpm-{task_name}-50epochs/checkpoints/{str(seed)}/last.ckpt"
    text2carb(checkpoint, model, carb, model.tokenizer, args.max_seq_length, args.train_batch_size, seed=seed)


if __name__ == "__main__":
    for task_name in ["econie", "oie2016"]:
        for seed in range(5):
            get_carb_output(task_name, seed)


