import nltk
import argparse
import json
from transformers import AutoTokenizer
from main import T5GenericFineTuner, get_generic_dataset, generate, get_carb_dataset, text2carb, object_generation_score

import pickle
import datasets

args_dict = dict(
    data_dir="",  # path for data files
    output_dir="",  # path to save the checkpoints
    model_name_or_path="",
    tokenizer_name_or_path="",
    max_seq_length=256,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=24,
    eval_batch_size=24,
    num_train_epochs=10,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=True,
    fp_16=False,  # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1',  # you can find out more on optimisation levels here
    # https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=0.5,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)

args = argparse.Namespace(**args_dict)


def combine_oie_ner():
    epoch, save_dir, checkpoint, model = 10, "", "/data/checkpoints/base-mpm-cased-ncner-60epochs.ckpt", "t5-base"

    model_name = model
    nltk.download('punkt')
    if 'small' in model_name:
        batch_size = 64
    elif 'base' in model_name:
        batch_size = 24
    elif 'large' in model_name:
        batch_size = 8
    else:
        print(model_name)
        raise ValueError("This model name is no currently supported.")
    MAX_LEN = 256

    args_dict = dict(
        data_dir="",  # path for data files
        output_dir=save_dir,  # path to save the checkpoints
        model_name_or_path=model_name,
        tokenizer_name_or_path=model_name,
        max_seq_length=MAX_LEN,
        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_train_epochs=epoch,
        gradient_accumulation_steps=16,
        n_gpu=1,
        early_stop_callback=True,
        fp_16=False,  # if you want to enable 16-bit training then install apex and set this to true
        opt_level='O1',  # you can find out more on optimisation levels here
        # https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=0.5,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
    )

    args = argparse.Namespace(**args_dict)
    ner_model = T5GenericFineTuner.load_from_checkpoint(checkpoint, task='ner', dataset_name='nc')
    oie_model = T5GenericFineTuner(args, "OIE", "econie")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ckpt = None

    # perform NER inference on OIE dataset
    train_dataset = get_generic_dataset(tokenizer, dataset_file=oie_model.dataset_file, type_path='train',
                                        prompt=ner_model.prompt, task=oie_model.task)
    train_ss, train_os, train_ts = generate(ckpt, ner_model, train_dataset, tokenizer, max_len=args.max_seq_length,
                                            batch_size=args.eval_batch_size)
    dev_dataset = get_generic_dataset(tokenizer, dataset_file=oie_model.dataset_file, type_path='validation',
                                      prompt=ner_model.prompt, task=oie_model.task)
    dev_ss, dev_os, dev_ts = generate(ckpt, ner_model, dev_dataset, tokenizer, max_len=args.max_seq_length,
                                      batch_size=args.eval_batch_size)
    test_dataset = get_generic_dataset(tokenizer, dataset_file=oie_model.dataset_file, type_path='test',
                                       prompt=ner_model.prompt, task=oie_model.task)
    test_ss, test_os, test_ts = generate(ckpt, ner_model, test_dataset, tokenizer, max_len=args.max_seq_length,
                                         batch_size=args.eval_batch_size)

    train = []
    dev = []
    test = []

    train_s = open("/data/t5-datasets/oie_ner/train.src", "w")
    train_t = open("/data/t5-datasets/oie_ner/train.tgt", "w")
    dev_s = open("/data/t5-datasets/oie_ner/dev.src", "w")
    dev_t = open("/data/t5-datasets/oie_ner/dev.tgt", "w")
    test_s = open("/data/t5-datasets/oie_ner/test.src", "w")
    test_t = open("/data/t5-datasets/oie_ner/test.tgt", "w")

    for s, o, t in zip(train_ss, train_os, train_ts):
        src = s[21:]
        tgt = ' '.join([f"({_o})" for _o in o.split('; ')]) + ' ' + t
        train_s.write(src + "\n")
        train_t.write(tgt + "\n")
        train.append({"source": src, "target": tgt})

    for s, o, t in zip(dev_ss, dev_os, dev_ts):
        src = s[21:]
        tgt = ' '.join([f"({_o})" for _o in o.split('; ')]) + ' ' + t
        dev_s.write(src + "\n")
        dev_t.write(tgt + "\n")
        dev.append({"source": src, "target": tgt})

    for s, o, t in zip(test_ss, test_os, test_ts):
        src = s[21:]
        tgt = ' '.join([f"({_o})" for _o in o.split('; ')]) + ' ' + t
        test_s.write(src + "\n")
        test_t.write(tgt + "\n")
        test.append({"source": src, "target": tgt})

    dataset_dict = datasets.DatasetDict({
        "train": datasets.Dataset.from_list(train),
        "validation": datasets.Dataset.from_list(dev),
        "test": datasets.Dataset.from_list(test)
    })
    with open("/data/t5-datasets/oie_ner.pkl", "wb") as f:
        pickle.dump(dataset_dict, f)


def carb_test_experiment():
    checkpoint = "/data/checkpoints/base-econie-10epochs.ckpt"
    oie_model = T5GenericFineTuner.load_from_checkpoint(checkpoint, task='OIE', dataset_name='econie')
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    carb_dataset = get_carb_dataset(tokenizer)
    text2carb(None, oie_model, carb_dataset, tokenizer, 256, 24, num_beams=4, printer="/data/carb/base-econie-beamsize-4.oie")
    text2carb(None, oie_model, carb_dataset, tokenizer, 256, 24, num_beams=2, printer="/data/carb/base-econie-beamsize-2.oie")
    text2carb(None, oie_model, carb_dataset, tokenizer, 256, 24, num_beams=1, printer="/data/carb/base-econie-beamsize-1.oie")

    checkpoint = "/data/checkpoints/base-mpm-econie-50epochs.ckpt"
    oie_model = T5GenericFineTuner.load_from_checkpoint(checkpoint, task='OIE', dataset_name='econie')
    text2carb(None, oie_model, carb_dataset, tokenizer, 256, 24, num_beams=4, printer="/data/carb/base-mpm-econie-beamsize-4.oie")
    text2carb(None, oie_model, carb_dataset, tokenizer, 256, 24, num_beams=2, printer="/data/carb/base-mpm-econie-beamsize-2.oie")
    text2carb(None, oie_model, carb_dataset, tokenizer, 256, 24, num_beams=1, printer="/data/carb/base-mpm-econie-beamsize-1.oie")

    checkpoint = "/data/checkpoints/flan-base-econie-10epochs.ckpt"
    oie_model = T5GenericFineTuner.load_from_checkpoint(checkpoint, task='OIE', dataset_name='econie')
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    carb_dataset = get_carb_dataset(tokenizer)
    text2carb(None, oie_model, carb_dataset, tokenizer, 256, 24, num_beams=4, printer="/data/carb/flan-base-econie-beamsize-4.oie")
    text2carb(None, oie_model, carb_dataset, tokenizer, 256, 24, num_beams=2, printer="/data/carb/flan-base-econie-beamsize-2.oie")
    text2carb(None, oie_model, carb_dataset, tokenizer, 256, 24, num_beams=1, printer="/data/carb/flan-base-econie-beamsize-1.oie")

    checkpoint = "/data/checkpoints/flan-base-mpm-econie-50epochs.ckpt"
    oie_model = T5GenericFineTuner.load_from_checkpoint(checkpoint, task='OIE', dataset_name='econie')
    text2carb(None, oie_model, carb_dataset, tokenizer, 256, 24, num_beams=4, printer="/data/carb/flan-base-mpm-econie-beamsize-4.oie")
    text2carb(None, oie_model, carb_dataset, tokenizer, 256, 24, num_beams=2, printer="/data/carb/flan-base-mpm-econie-beamsize-2.oie")
    text2carb(None, oie_model, carb_dataset, tokenizer, 256, 24, num_beams=1, printer="/data/carb/flan-base-mpm-econie-beamsize-1.oie")


def ncner_dataset():
    with open("/mnt/c/users/hyun1/Downloads/eng-ner-test-3000-org-split-230118.json") as f:
        test = json.load(f)
    
    with open("/mnt/c/users/hyun1/Downloads/eng-ner-train-20100-org-split-230118.json") as f:
        train = json.load(f)

    def simple_tokenize(text):
        tokens = []
        current_char = ""
        for char in text:
            if char.isalnum():
                current_char += char
            elif char == " ":
                if current_char:
                    tokens.append(current_char)
                current_char = ""
            else: # special char
                if current_char:
                    tokens.append(current_char)
                tokens.append(char)
                current_char = ""
        if current_char:
            tokens.append(current_char)
        return tokens

    def output_tokens(example):
        text = example['text']
        idx = 0
        target_text = ""
        target_tokens = []
        annotations = sorted(example['annotations'], key=lambda k: k['start'])
        for ne in annotations:
            target_text += text[idx: ne['start']]
            target_tokens.extend(simple_tokenize(text[idx: ne['start']]))

            label = ne['label'].split('_')[0]
            target_text += ("B-" + label)
            target_tokens.append("B-" + label)
            for word in simple_tokenize(ne['text'])[1:]:
                target_text += (" I-" + label)
                target_tokens.append("I-" + label)
            idx = ne['end']
        target_text += text[idx:]
        target_tokens.extend(simple_tokenize(text[idx:]))
        return target_text, target_tokens

    def preprocess(split):
        processed_split = []
        for ex in split:
            source_text = ex['text']
            source_tokens = simple_tokenize(source_text)
            target_text, target_tokens = output_tokens(ex)
            if len(source_tokens) != len(target_tokens):
                continue
            processed_split.append({
                    "source_text": source_text,
                    "source_tokens": source_tokens,
                    "target_text": target_text,
                    "target_tokens": target_tokens
                })
        return processed_split

    p_train = preprocess(train)
    p_test = preprocess(test)

    dataset_dict = datasets.DatasetDict({"train": datasets.Dataset.from_list(p_train),
                                        "test": datasets.Dataset.from_list(p_test)})

    with open("/mnt/e/s4_dataset/ncner-enc.pkl", 'wb') as f:
        pickle.dump(dataset_dict, f)


def eval_ner_models():
    for ckpt in ["base-tacred-3epochs.ckpt", "base-mpm-tacred-43epochs.ckpt",
              "flan-base-mpm-tacred-43epochs.ckpt"]:
        if 'flan' in ckpt:
            args.model_name_or_path = "google/flan-t5-base"
            args.tokenizer_name_or_path = "google/flan-t5-base"
        else:
            args.model_name_or_path = "t5-base"
            args.tokenizer_name_or_path = "t5-base"
        checkpoint = "/data/checkpoints/" + ckpt
        model = T5GenericFineTuner.load_from_checkpoint(checkpoint, task="ORE", dataset_name="tacred")
        test_dataset = get_generic_dataset(model.tokenizer, model.dataset_file, 'test', model.prompt, model.task)
        texts, outputs, targets = generate(None, model, test_dataset, model.tokenizer, args.eval_batch_size)
        object_generation_score(texts, outputs, targets)

if __name__ == "__main__":
    # carb_test_experiment()
    # combine_oie_ner()
    # eval_ner_models()
    ncner_dataset()
