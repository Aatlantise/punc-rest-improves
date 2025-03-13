import random
import pickle
import time
import argparse
import os

from tqdm import tqdm
import numpy as np
from typing import Optional

from main import T5GenericFineTuner, T5TokenClassificationDataset, T5ClassificationHead

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import AdamW, AutoTokenizer, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput, Seq2SeqSequenceClassifierOutput, TokenClassifierOutput
from transformers.models.t5.modeling_t5 import T5Config

from sklearn.metrics import classification_report


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class OptimizedT5EncoderForTokenClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.num_labels = 9
        config.learning_rate = 3e-4
        config.weight_decay = 0.0
        config.adam_epsilon = 1e-8
        config.classifier_dropout = 0.1
        self.transformer = AutoModel.from_pretrained(config.model_name, config=config)
        self.classification_head = T5ClassificationHead(config)
        self.max_input_words = 256

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.decoder_device_map = None
        self.parallelize()

        # initialize
        self.optimizer = None
        # self.transformer.post_init()
        self.configure_optimizers()

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.transformer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.transformer.config.weight_decay,
            },
            {
                "params": [p for n, p in self.transformer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.transformer.config.learning_rate, eps=self.transformer.config.adam_epsilon)
        self.optimizer = optimizer
        return [optimizer]

    def parallelize(self):
        self.transformer.encoder.parallelize()
        self.device_map = self.transformer.device_map
        self.classification_head.to("cuda")
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.encoder.deparallelize()
        self.classification_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                labels=None
                ):
        # Get the encoder's hidden states

        encoder_output = self.transformer.encoder(input_ids.to("cuda"), attention_mask=attention_mask)
        logits = self.classification_head(encoder_output.last_hidden_state)

        # here we get (64, 256, 9) but want (64, 9, 256)
        logits = logits.movedim(1, -1)

        return logits


class BaselineT5EncoderForTokenClassification(nn.Module):
    def __init__(self, t5_model):
        super().__init__()
        self.t5_model = t5_model
        self.t5_model.to("cuda")
        # self.classification_head = nn.Linear(self._get_encoder().config.d_model, 9)
        # self.classification_head.to("cuda")

        config = T5Config.from_pretrained("t5-base")
        config.num_labels = 9
        config.learning_rate = 3e-4
        config.weight_decay = 0.0
        config.adam_epsilon = 1e-8
        config.classifier_dropout = 0.1

        self.classification_head = T5ClassificationHead(config)
        self.configure_optimizers()
        self.max_input_words = 256

    def _get_encoder(self):
        return self.t5_model.model.encoder

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.t5_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.t5_model.hparam.weight_decay,
            },
            {
                "params": [p for n, p in self.t5_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.t5_model.hparam.learning_rate, eps=self.t5_model.hparam.adam_epsilon)
        self.optimizer = optimizer
        return [optimizer]

    def forward(self, input_ids, attention_mask, labels=None):
        # Get the encoder's hidden states
        encoder_output = self._get_encoder()(input_ids, attention_mask=attention_mask)

        # For NER, we don't want to pool
        logits = self.classification_head(encoder_output.last_hidden_state)

        # here we get (64, 256, 9) but want (64, 9, 256)
        logits = logits.movedim(1, -1)

        return logits


def train(model, args):
    dataloader = args.dataloader

    message = "USL training" if args.mpm else "Vanilla training"
    for epoch in range(args.num_epochs):
        pbar = tqdm(dataloader, desc=message)
        print(f"Training epoch {epoch}...")
        for batch in pbar:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            # Forward pass, we need (64, 9, 256)
            logits = model(input_ids=input_ids.to("cuda"),
                           attention_mask=attention_mask.to("cuda"),
                           labels=labels.to("cuda"))
            if type(logits) in [TokenClassifierOutput, Seq2SeqSequenceClassifierOutput, SequenceClassifierOutput]:
                logits = logits.logits
            if logits.shape == (64, 256, 9):
                logits = logits.movedim(1, -1)

            # Compute loss (e.g., cross-entropy)
            m = nn.CrossEntropyLoss(reduction='none')
            _loss = m(logits.to("cuda"), labels.to("cuda"))
            attention_mask = attention_mask.to("cuda")
            loss = (_loss * attention_mask).sum() / attention_mask.sum()

            # Backpropagation and optimization
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            pbar.set_description(f"{message} (Loss: {loss:.4f})")

    return model


def enct5_train(model, args):
    """
    Performs training on a OptimizedT5Encoder model. Use the following code:

    args.mpm = False
    config = T5Config.from_pretrained(args.model_name_or_path)
    model = OptimizedT5EncoderForTokenClassification(config)
    model = enct5_train(model, args)
    p, r, f1 = ner_eval(model)

    args.mpm = True
    usl_model = OptimizedT5EncoderForTokenClassification(config)
    model = enct5_train(usl_model, args)
    p, r, f1 = ner_eval(usl_model)

    :param model:
    :param args:
    :return:
    """
    if args.mpm and 'flan' in args.model_name_or_path:
        model.transformer.load_state_dict(torch.load("/data/checkpoints/flan-t5-base-mpm.pt"))
    elif args.mpm:
        model.transformer.load_state_dict(torch.load("/data/checkpoints/t5-base-mpm.pt"))

    ncner = pickle.load(open(args.dataset, "rb"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    train_dataset = T5TokenClassificationDataset(tokenizer, ncner, 'train')
    dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=8,
                            drop_last=True)
    args.dataloader = dataloader

    return train(model, args)


def baseline_train(args):
    """
    Performs training on a T5GenericFineTuner model. Use the following code:

    args.mpm = False
    vanilla_model = baseline_train(args)
    p, r, f1 = ner_eval(vanilla_model)

    args.mpm = True
    usl_model = baseline_train(args)
    p, r, f1 = ner_eval(usl_model)
    """
    if args.mpm:
        T5GenericFineTuner._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        t5_model = T5GenericFineTuner.load_from_checkpoint("/data/t5-mpm-checkpoints/base-mpm-40epochs.ckpt",
                                                           task='ner', dataset_name='nc', strict=False)
    else:
        T5GenericFineTuner._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        t5_model = T5GenericFineTuner(args, task='ner', dataset_name='nc')
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    s4e_model = BaselineT5EncoderForTokenClassification(t5_model)
    s4e_model.tokenizer = tokenizer

    s4e_model = s4e_model.to("cuda")

    tokenizer = s4e_model.tokenizer
    ncner = pickle.load(open("/data/t5-datasets/ncner-enc.pkl", "rb"))
    train_dataset = T5TokenClassificationDataset(tokenizer, ncner, 'train')
    dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=8)

    args.dataloader = dataloader

    return train(s4e_model, args)


def parse_labels(flattened_entities):
    entities = []
    entity = ()
    for i, g in enumerate(flattened_entities):
        if g == 0:  # nothing to see here, check if in middle of entity
            if len(entity) == 1:  # single length entity
                entity = (entity[0], i)
                entities.append(entity)
                entity = ()
            elif len(entity) == 2:
                entities.append(entity)
                entity = ()
            else:
                continue
        elif g in [1, 3, 5, 7]:  # start of new entity
            if len(entity) == 1:
                entity = (entity[0], i)
                entities.append(entity)
                entity = (i,)
            elif len(entity) == 2:
                entities.append(entity)
                entity = (i,)
            else:  # no entity
                entity = (i,)
        else:  # middle of entity
            if len(entity) == 0:
                # some kind of error...
                continue
            elif len(entity) == 1:
                entity = (entity[0], i)
            elif len(entity) == 2:
                entity = (entity[0], i)
    return entities


def ner_eval(finetuned_model, batch_size=256):

    finetuned_model.eval()
    # tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    print(f"Model loaded...")

    ncner = pickle.load(open("/data/t5-datasets/ncner-enc.pkl", "rb"))
    test_dataset = T5TokenClassificationDataset(tokenizer, ncner, 'test')
    print("Dataset loaded...")

    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    all_predictions = []
    all_labels = []

    start_time = time.time()

    print("Evaluating...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            true_labels = batch["labels"].to("cuda")  # Replace with your labels

            # Forward pass
            logits = finetuned_model(input_ids, attention_mask)

            if type(logits) in [TokenClassifierOutput, Seq2SeqSequenceClassifierOutput, SequenceClassifierOutput]:
                logits = logits.logits
            if logits.shape == (batch_size, 256, 9):
                logits = logits.movedim(1, -1)

            m = nn.Softmax(dim=1)
            acc = m(logits)
            pred_labels = torch.argmax(acc, dim=1)

            gold_labels_flat = true_labels.flatten()
            pred_labels_flat = pred_labels.flatten()

            all_predictions.extend(pred_labels_flat.to("cpu"))
            all_labels.extend(gold_labels_flat.to("cpu"))

    num_sents = len(dataloader) * batch_size
    time_took = time.time() - start_time

    report = classification_report(all_labels, all_predictions)
    print(report)

    gold_entities = parse_labels(all_labels)
    pred_entities = parse_labels(all_predictions)

    gold = len(gold_entities)
    attempt = len(pred_entities)

    correct = len(list(set(gold_entities) & set(pred_entities)))

    print(f"Prediction on {num_sents} sentences took {time_took} seconds")
    print(f"That is {num_sents/time_took} sentences per second")

    p = correct / attempt
    r = correct / gold
    f1 = 2 * p * r / (p + r)
    print(f"From a corpus of {gold} problems, {attempt} attempts were made. \n"
          f"Precision: {p}, recall: {r}, f1: {f1} \n"
          )

    return p, r, f1


def main(args):
    f = open("experiments.log", "w")

    for i in range(args.num_runs):
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(i)
        random.seed(i)

        config = T5Config.from_pretrained(args.model_name_or_path)
        config.model_name = args.model_name_or_path

        args.mpm = False
        vanilla_model = OptimizedT5EncoderForTokenClassification(config)
        vanilla_model = enct5_train(vanilla_model, args)
        p, r, f1 = ner_eval(vanilla_model, batch_size=args.eval_batch_size)
        if f1 > 0.9:
            args.model_name_or_path = "flan-t5-base" if 'flan' in args.model_name_or_path else args.model_name_or_path
            torch.save(vanilla_model.state_dict(), f"lightning_logs/{args.model_name_or_path}-encoder-ner-f1-%.2f.pt" % f1)
        f.write("\t".join([str(p), str(r), str(f1)]) + "\t")

        args.mpm = True
        usl_model = OptimizedT5EncoderForTokenClassification(config)
        usl_model = enct5_train(usl_model, args)
        p, r, f1 = ner_eval(usl_model, batch_size=args.eval_batch_size)
        if f1 > 0.9:
            args.model_name_or_path = "flan-t5-base" if 'flan' in args.model_name_or_path else args.model_name_or_path
            torch.save(usl_model.state_dict(), f"lightning_logs/{args.model_name_or_path}-encoder-usl-ner-f1-%.2f.pt" % f1)
        f.write("\t".join([str(p), str(r), str(f1)]) + "\n")


if __name__ == "__main__":
    args_dict = dict(
        model_name_or_path="t5-base",
        dataset="/data/t5-datasets/ncner-enc.pkl",
        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        gradient_accumulation_steps=1,
        n_gpu=1,
        fp_16=False,
        num_epochs=10,
        max_grad_norm=0.5,
        train_batch_size=128,
        eval_batch_size=128,
        max_seq_length=256,
        warmup_steps=0,
        mpm=False,
        seed=42,
        num_labels=9,
        num_runs=120
    )
    args = argparse.Namespace(**args_dict)

    main(args)
