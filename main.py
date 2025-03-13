import logging
import torch
import os
import pickle
import random

import pytorch_lightning as pl
import numpy as np

from tqdm import tqdm
from typing import List, Any, Dict,  Union, Optional
from torch.utils.data import Dataset, DataLoader
from torch import autograd, nn
from transformers.models.t5.modeling_t5 import T5Config
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    T5TokenizerFast,
    get_linear_schedule_with_warmup,
    GPT2Tokenizer,
    GPT2LMHeadModel
)

autograd.set_detect_anomaly(True)
logger = logging.getLogger(__name__)


def flatten(nested_list: List[List[Any]]) -> List[Any]:
    """
    Flattens nested list

    :param nested_list: nested list of rank 2
    :return: flattened list
    """
    flat_list = []
    for sublist in nested_list:
        flat_list.extend(sublist)
    return flat_list


def set_seed(seed):
    """
    Sets seed to given number, used to control seed in multi-seed evaluation

    :param seed:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_generic_dataset(tokenizer, dataset_file, type_path, prompt, task, max_len=256, cased=True, padding='max_length'):
    """
    Returns generic dataset object

    :param tokenizer: Tokenizer
    :param dataset_file: Pickled Huggingface datasets file
    :param type_path: train, validation, or test?
    :param prompt: Prompt to prepend to source sentence
    :param task: Task that this dataset represents--used to determine evaluator
    :param max_len: Maximum length sequence, defaults to 256
    :param cased: Casing, defaults to True
    :param padding: Padding options, defaults to max_length
    :return: GenericDataset object
    """
    assert type_path in ["train", "validation", "test"]
    assert task.lower() in ["ner", "oie", "sbd", "chunk", "pos", "srl", "ore", "mpm", "ko_usl"]
    dataset = pickle.load(open(dataset_file, 'rb'))
    if "validation" not in dataset:
        dataset['validation'] = dataset['test']
    return GenericDataset(tokenizer=tokenizer, dataset=dataset, type_path=type_path, prompt=prompt, task=task,
                          max_len=max_len, cased=cased, padding=padding)


def get_gold_dataset(tokenizer):
    """
    Returns OpenIE's Handcrafted Gold dataset

    :param tokenizer: Tokenizer
    :return: GoldDataset object
    """
    return GoldDataset(tokenizer=tokenizer)


def get_carb_dataset(tokenizer):
    """
    Returns CaRB evaluation dataset

    :param tokenizer: Tokenizer
    :return: CaRBDataset object
    """
    return CarbDataset(tokenizer=tokenizer)


class T5Dataset(Dataset):
    """
    T5Dataset object. Inherits torch.utils.data.Dataset class. Is the mother class for GoldDataset (handcrafted gold
    OpenIE dataset), CaRBDataset (OpenIE CaRB dataset), and the GenericDataset (all other datasets) classes.
    """
    def __init__(self, tokenizer, max_len=256):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.labels = []

    def __len__(self):
        return len(self.inputs)

    @staticmethod
    def _compare_source_target_tokens(s_tokens, t_tokens):
        return [(n, s, t) for n, (s, t) in enumerate(zip(s_tokens, t_tokens)) if
                t.lower().replace(".", "").replace(",", "").replace("'", "").replace('"', '') != s]

    def __getitem__(self, index):

        source_ids = self.inputs[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze

        if self.targets:
            target_ids = self.targets[index]["input_ids"].squeeze()
            target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze
            return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids,
                    "target_mask": target_mask}
        elif self.labels:  # labels exist
            labels = self.labels[index]
            labels = torch.from_numpy(np.array(labels, dtype=np.int))
            return {"input_ids": source_ids, "attention_mask": src_mask, "labels": labels}
        else:
            return {"input_ids": source_ids, "attention_mask": src_mask, }


class GenericDataset(T5Dataset):
    """
    Inherits the T5Dataset class, used for additional pre-training and fine-tuning
    """
    def __init__(self, tokenizer, dataset, type_path, prompt, task, max_len=256, cased=True, padding='max_length'):
        """
        Initializes the GenericDataset class

        :param tokenizer: Tokenizer
        :param dataset_file: Pickled Huggingface datasets file
        :param type_path: train, validation, or test?
        :param prompt: Prompt to prepend to source sentence
        :param task: Task that this dataset represents--used to determine evaluator
        :param max_len: Maximum length sequence, defaults to 256
        :param cased: Casing, defaults to True
        :param padding: Padding options, defaults to max_length
        """
        super().__init__(tokenizer, max_len)
        self.data = dataset[type_path]
        self.prompt = prompt
        self.cased = cased
        if self.prompt[-2:] == ": ":
            pass
        elif self.prompt[-1] == ":":
            self.prompt += " "
        else:
            self.prompt += ": "
        self.task = task
        self.padding = padding
        self._build()

    def _build(self):
        """
        Pre-processes and tokenize training data.

        Pre-processing includes the following steps:
        1. If task is punctuation restoration (MPM), remove punctuation
        2. If casing is True, remove casing
        3. Prepend prompt, add EOS token </s>

        :return:
        """
        for idx in range(len(self.data)):
            if self.task.lower() == "mpm":
                target = self.data[idx]['text']
                input_ = target.replace('.', '').replace(',', '').replace('"', '').replace("'", "")
            else:
                input_, target = self.data[idx]["source"], self.data[idx]["target"]

            if self.cased:
                pass
            else:
                input_ = input_.lower()
                target = target.lower()

            input_ = self.prompt + input_ + ' </s>'
            target = target + " </s>"

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=self.max_len, padding=self.padding, truncation=True, return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len, padding=self.padding, truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)


class CarbDataset(T5Dataset):
    """
    CaRBDataset class, whose object instance is returned by the get_carb_dataset method.
    The CaRB OpenIE dataset is used for evaluation only.
    """
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.data = [k.strip("\n") for k in open("/data/carb/carb_sentences.txt")]
        self.inputs = []
        self._build()

    def _build(self):
        for idx in range(len(self.data)):
            input_ = self.data[idx] + " </s>"

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)


class GoldDataset(T5Dataset):
    """
    GoldDataset class, whose object instance is returned by the get_gold_dataset method.
    The handcrafted gold OpenIE dataset is used for evaluation only.
    """
    def __init__(self, tokenizer, max_len=256):
        super().__init__(tokenizer, max_len)
        self.filename = '/data/gold/gold.txt'
        self.data = []
        prev_sent = ""
        with open(self.filename, encoding='unicode_escape') as f:
            # check if new sent
            target = []
            for line in f:
                sent, head, pred, tail = line.strip('\n').split('\t')[:4]

                # current sent equals previous sent: add to target
                if sent == prev_sent:
                    target.append(f"({head}; {pred}; {tail})")
                    prev_sent = sent

                # new sent: add to data and re-initialize
                else:
                    if target:
                        self.data.append({"source": prev_sent, "target": ' '.join(target)})
                    else:
                        pass
                    prev_sent = sent
                    target = [f"({head}; {pred}; {tail})"]

        self._build()

    def _build(self):
        for idx in range(len(self.data)):
            input_, target = self.data[idx]["source"], self.data[idx]["target"]

            input_ = 'Extract event or relation triples: ' + input_ + ' </s>'
            target = target + " </s>"

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)


class T5ClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks. Source code from
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
    """

    def __init__(self, config: T5Config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class T5TokenClassificationDataset(T5Dataset):
    """
    Inherits the T5Dataset class, to be used for token classification tasks like NER.
    """
    def __init__(self, tokenizer, dataset, type_path, max_len=256, task='NER'):
        super().__init__(tokenizer, max_len)
        self.data = dataset[type_path]
        self.labels = []
        self.text = []
        self.task = task
        self._build()

    def _build(self):
        for idx in tqdm(range(0, len(self.data), 32)):
            slice = self.data[idx: idx + 32]

            if self.task == 'MNLI':
                inputs = [slice['premise'][i] + ' <SEP> ' + slice['hypothesis'][i] + ' </s>' for i in range(len(slice['premise']))]
                labels = slice['label']

            else: # NER
                token2label = {"B-PER": 1, "I-PER": 2,
                               "B-LOC": 3, "I-LOC": 4,
                               "B-ORG": 5, "I-ORG": 6,
                               "B-PROD": 7, "I-PROD": 8}
                inputs = [s for s in slice['source_tokens']]
                word_labels = []
                for s in slice['target_tokens']:
                    word_label = [token2label.get(t, 0) for t in s]
                    word_labels.append(word_label)

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                inputs, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt",
                is_split_into_words=(self.task == 'NER') # True if NER
            )

            # if NER, align labels, accounting for subword tokens
            if self.task == 'NER':
                labels = []
                for i, label in enumerate(word_labels):
                    word_ids = tokenized_inputs.word_ids(batch_index=i)
                    previous_word_idx = None
                    label_ids = []
                    first_none = True
                    for word_idx in word_ids:
                        if word_idx is None:
                            if first_none:
                                first_none = False
                                label_ids.append(0)  # end of sent here, account for </s> token, adding corresponding label
                            label_ids.append(-100)
                        elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                            label_ids.append(label[word_idx])
                        else:
                            # label_ids.append(-100)
                            label_ids.append(label[word_idx])
                        previous_word_idx = word_idx
                    label_ids.extend([-100] * self.max_len)
                    label_ids = label_ids[:self.max_len]
                    labels.append(label_ids)

            self.inputs.extend([{"input_ids": tokenized_inputs["input_ids"][i],
                                 "attention_mask": tokenized_inputs["attention_mask"][i]} for i in range(len(labels))])
            self.labels.extend(labels)
            self.text.extend(inputs)


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparam):
        # TODO: separate
        super(T5FineTuner, self).__init__()
        self.hparam = hparam
        if 'gpt' in hparam.model_name_or_path.lower():
            self.model = GPT2LMHeadModel.from_pretrained(hparam.model_name_or_path)
            self.tokenizer = GPT2Tokenizer.from_pretrained(hparam.model_name_or_path)
        elif 'mt5' in hparam.model_name_or_path.lower():
            self.model = MT5ForConditionalGeneration.from_pretrained(hparam.model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(
                hparam.model_name_or_path
            )
        elif 'paust' in hparam.model_name_or_path.lower():
            self.model = T5ForConditionalGeneration.from_pretrained(
                hparam.model_name_or_path)
            self.tokenizer = T5TokenizerFast.from_pretrained(hparam.model_name_or_path)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(
                hparam.model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(
                hparam.model_name_or_path
            )
        self.save_hyperparameters()
        self.lr_scheduler = None

    def is_logger(self):
        return True

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        self.log("train_loss", avg_train_loss)

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)
        tensorboard_logs = {"val_loss": avg_loss}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparam.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparam.learning_rate, eps=self.hparam.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self,
                       epoch=None,
                       batch_idx=None,
                       optimizer=None,
                       optimizer_idx=None,
                       optimizer_closure=None,
                       on_tpu=None,
                       using_native_amp=None,
                       using_lbfgs=None
                       ):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(
            self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict


class T5GenericFineTuner(T5FineTuner):
    def __init__(self, hparam, task, dataset_name, epoch_end_eval=True, cased=True):
        super().__init__(hparam)
        self.task = task
        self.hparam.dataset = self.task
        self.cased = cased
        self.epoch_end_eval = epoch_end_eval

        if self.task.lower() == "ner":
            self.prompt = "Find named entities: "
            self.evaluator = object_generation_score
            if dataset_name.lower() == "nc":
                self.dataset_file = "/data/t5-datasets/nc-ner.pkl"
            elif dataset_name.lower() == "hyun":
                self.dataset_file = "/data/t5-datasets/hyun_ner.pkl"
            elif dataset_name.lower() == "genia":
                self.dataset_file = "/data/t5-datasets/genia.ner.pkl"
            elif dataset_name.lower() == "conll_2003":
                self.dataset_file = "/data/t5-datasets/conll_03_ner_gen.pkl"
            elif dataset_name.lower() == "ontonotes":
                self.dataset_file = "/data/t5-datasets/ontonotes.ner.pkl"
            else:
                raise ValueError(f"Dataset {dataset_name} not supported")

        elif self.task.lower() == "oie":
            self.prompt = "Extract event or relation triples: "
            self.evaluator = object_generation_score

            if dataset_name.lower() == "econie":
                self.dataset_file = "/data/t5-datasets/econie3_nolabels.pkl"
            elif dataset_name.lower() == "openie4":
                self.dataset_file = "/data/t5-datasets/openie4_labels.pkl"
            elif dataset_name.lower() == "oie2016":
                self.dataset_file = "/data/t5-datasets/oie2016.pkl"
            elif dataset_name.lower() == "oie_ner":
                self.dataset_file = "/data/t5-datasets/oie_ner.pkl"
                self.prompt = "Extract event or relation triples and named entities: "
                self.evaluator = multitask_score

        elif self.task.lower() == "sbd":
            self.prompt = "Detect sentence boundaries: "
            self.dataset_file = "/data/t5-datasets/ptb_sbd.pkl"
            self.evaluator = sbd_score
            self.tokenizer = T5Tokenizer.from_pretrained(hparam.model_name_or_path,
                                                         additional_special_tokens=["<EOS>"],
                                                         extra_ids=0)

        elif self.task.lower() == "chunk":
            self.prompt = "Perform chunking: "
            self.evaluator = sequence_tagging_score
            if dataset_name.lower() == "conll_2000":
                self.dataset_file = "/data/t5-datasets/conll00_chunk.pkl"
            elif dataset_name.lower() == "conll_2003":
                self.dataset_file = "/data/t5-datasets/conll_03_chunk.pkl"

        elif self.task.lower() == "pos":
            self.prompt = "Identify parts of speech: "
            self.evaluator = sequence_tagging_score
            if dataset_name.lower() == "conll_2000":
                self.dataset_file = "/data/t5-datasets/conll00_pos.pkl"
            elif dataset_name.lower() == "conll_2003":
                self.dataset_file = "/data/t5-datasets/conll_03_pos.pkl"

        elif self.task.lower() == "srl":
            self.prompt = "Identify semantic roles: "
            self.dataset_file = "/data/t5-datasets/conll12_srl.pkl"
            self.evaluator = object_generation_score

        elif self.task.lower() == "ore":
            self.prompt = "Classify relations: "
            self.dataset_file = "/data/t5-datasets/tacred.pkl"
            self.evaluator = object_generation_score
            self.tokenizer = T5Tokenizer.from_pretrained(hparam.model_name_or_path,
                                                         additional_special_tokens=["<e1>", "</e1>",
                                                                                    "<e2>", "</e2>"],
                                                         extra_ids=0)

        elif self.task.lower() == "mpm":
            self.prompt = "Add punctuation as appropriate:"
            self.dataset_file = "/data/t5-datasets/es_text437031.pkl"
            self.evaluator = mpm_score

        elif self.task.lower() == "ko_usl":
            self.prompt = "다음 문장에서 문장부호를 복구하세요:"
            self.dataset_file = "/data/t5-datasets/ko_usl_data.pkl"
            self.evaluator = mpm_score

    def train_dataloader(self):
        _args = {"tokenizer": self.tokenizer, "dataset_file": self.dataset_file, "type_path": "train",
                 "prompt": self.prompt, "task": self.task,
                 "max_len": self.hparam.max_seq_length,
                 "padding": "do_not_pad" if 'gpt' in self.hparam.model_name_or_path.lower() else 'max_length'}
        if not self.cased:
            _args['cased'] = False
        train_dataset = get_generic_dataset(**_args)
        dataloader = DataLoader(train_dataset, batch_size=self.hparam.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=8)
        t_total = (
                (len(dataloader.dataset) //
                 (self.hparam.train_batch_size * max(1, self.hparam.n_gpu)))
                // self.hparam.gradient_accumulation_steps
                * float(self.hparam.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparam.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        _args = {"tokenizer": self.tokenizer, "dataset_file": self.dataset_file, "type_path": "validation",
                 "prompt": self.prompt, "task": self.task,
                 "max_len": self.hparam.max_seq_length,
                 "padding": "do_not_pad" if 'gpt' in self.hparam.model_name_or_path.lower() else 'max_length'}
        if not self.cased:
            _args['cased'] = False
        val_dataset = get_generic_dataset(**_args)
        return DataLoader(val_dataset, batch_size=self.hparam.eval_batch_size, num_workers=8)

    def test_dataloader(self):
        _args = {"tokenizer": self.tokenizer, "dataset_file": self.dataset_file, "type_path": "test",
                 "prompt": self.prompt, "task": self.task,
                 "max_len": self.hparam.max_seq_length,
                 "padding": "do_not_pad" if 'gpt' in self.hparam.model_name_or_path.lower() else 'max_length'}
        if not self.cased:
            _args['cased'] = False
        test_dataset = get_generic_dataset(**_args)

        return DataLoader(test_dataset, batch_size=self.hparam.eval_batch_size, num_workers=8)

    def validation_epoch_end(self, outputs):
        if self.epoch_end_eval and self.current_epoch < 1:
            self.log("val_f1", 0.00)
        if self.epoch_end_eval and self.current_epoch >= 1:
            self.model.eval()

            with torch.no_grad():
                # initialize models
                test_dataloader = self.test_dataloader()

                # loop through dataloader
                _outputs = []
                targets = []
                texts = []
                for batch in tqdm(test_dataloader):
                    outs = self.model.generate(input_ids=batch['source_ids'].to("cuda"),
                                               attention_mask=batch['source_mask'].to("cuda"),
                                               max_length=self.hparam.max_seq_length)
                    dec = [
                        self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                        for ids in outs]
                    target = [
                        self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                        for ids in batch["target_ids"].to("cuda")]
                    text = [
                        self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                        for ids in batch["source_ids"].to("cuda")]
                    texts.extend(text)
                    _outputs.extend(dec)
                    targets.extend(target)

            # return model to training mode
            self.model.train()

            metric = self.evaluator(texts, _outputs, targets)
            self.log("val_f1", metric)
        else:
            avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
            self.log("val_loss", avg_loss)



class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))


class EncT5Tokenizer(T5Tokenizer):
    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        additional_special_tokens=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        super().__init__(
            vocab_file=vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            sp_model_kwargs=sp_model_kwargs,
            **kwargs,
        )

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # normal case: some special tokens
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make
        use of token type ids, therefore a list of zeros is returned.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of zeros.
        """
        bos = [self.bos_token_id]
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(bos + token_ids_0 + eos) * [0]
        return len(bos + token_ids_0 + eos + token_ids_1 + eos) * [0]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:
        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s> B </s>`
        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        else:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]


def run(args, model, resume_from_checkpoint=None, monitor="val_f1", seed=42):
    if seed == "random":
        seed = random.randint(0, 1000)
    set_seed(seed)

    if monitor == "val_f1":
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            filename='{epoch}-{val_f1:.3f}',
            monitor="val_f1", mode="max", save_top_k=2, every_n_epochs=1, save_last=True
        )
    else:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            filename='{epoch}-{val_loss:.3f}',
            monitor="val_loss", mode="min", save_top_k=2, every_n_epochs=1, save_last=True
        )

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        # early_stop_callback=False,
        precision=16 if args.fp_16 else 32,
        # amp_level=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
        # checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback(), checkpoint_callback],
    )

    trainer = pl.Trainer(**train_params)
    trainer.fit(model, ckpt_path=resume_from_checkpoint)

    return model


def clean_split(s: str) -> List[str]:
    return [k.strip(' ') for k in s.strip(' ').strip(')').strip('(').strip(' ').split(') (')]


def generate(ckpt: Union[str, None], model, input_dataset, tokenizer, batch_size, max_len=256, num_beams=4, skip_special_tokens=True, shuffle=True):
    if ckpt is not None:
        model = model.load_from_checkpoint(ckpt)
    dataloader = DataLoader(input_dataset, batch_size=batch_size, num_workers=8, shuffle=shuffle)
    model.model.eval()
    model = model.to("cuda")

    # loop through dataloader
    outputs = []
    targets = []
    texts = []
    for batch in tqdm(dataloader):
        outs = model.model.generate(input_ids=batch['source_ids'].to("cuda"),
                                    attention_mask=batch['source_mask'].to("cuda"),
                                    max_length=max_len, num_beams=num_beams
                                    )
        dec = [tokenizer.decode(ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False).strip() for ids in
               outs]
        target = [tokenizer.decode(ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False).strip()
                  for ids in batch["target_ids"]]
        text = [tokenizer.decode(ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False).strip()
                for ids in batch["source_ids"]]
        texts.extend(text)
        outputs.extend(dec)
        targets.extend(target)

    return texts, outputs, targets


def text2triple(outputs, targets):
    output_list = []
    target_list = []
    for output, target in zip(outputs, targets):

        sentence_outputs = []
        sentence_targets = []

        if output == "":
            pass
        else:
            _os = ['(' + k.strip(')').strip('(') + ')' for k in output.split(') (')]
            for o in _os:
                output_split = o.split(';')
                if len(output_split) < 3:
                    output_split.extend(["", "", ""])
                head, pred, tail = output_split[:3]
                sentence_outputs.append({
                    "head": head.replace("(", "").strip(' '),
                    "predicate": pred.strip(' '),
                    "tail": tail.replace(")", "").strip(' ')
                })

        if target == "":
            pass
        else:
            ts = ['(' + k.strip(' ').strip(')').strip('(') + ')' for k in target.split(') (')]
            for t in ts:
                target_split = t.split(';')
                if len(target_split) < 3:
                    target_split.extend(["", "", ""])
                head, pred, tail = target_split[:3]
                sentence_targets.append({
                    "head": head.replace("(", "").strip(' '),
                    "predicate": pred.strip(' '),
                    "tail": tail.replace(")", "").strip(' ')
                })

        output_list.append(sentence_outputs)
        target_list.append(sentence_targets)

    return output_list, target_list


def sequence_tagging_score(texts, outputs, targets, printer=print):
    """
    Calculates sequence tagging score
    :param outputs: Inferred outputs
    :param targets: Gold labels
    :return: Weighted sequence tagging score
    """
    gold = 0
    correct = 0
    attempt = 0
    for output, target in zip(outputs, targets):
        os = output.split(' ')
        ts = target.split(' ')

        if len(ts) >= len(os):
            ts = ts[:len(os)]
        else:
            ts.extend(['O'] * (len(os) - len(ts)))

        assert len(os) == len(ts)

        for o, t in zip(os, ts):
            if o != 'O':
                gold += 1
                if o == t:
                    correct += 1
            if t != 'O':
                attempt += 1

    precision = correct / gold
    recall = correct / attempt
    printer(f"Correct: {correct}, gold: {gold}: precision {'%.3f' % precision}\n")
    printer(f"Correct: {correct}, attempts: {attempt}: recall {'%.3f' % recall}\n ")

    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    printer(f"F1 score: {'%.3f' % f1} \n")

    for i in range(3):
        printer("Input sentence:     %s \n" % texts[i])
        printer("Actual sentence:    %s \n" % targets[i])
        printer("Predicted sentence: %s \n" % outputs[i])
        printer("=====================================================================\n")

    return f1


def object_generation_score(texts, outputs, targets, printer=print):
    # initialize metrics
    attempts = 0
    total_gold = 0
    correct = 0

    for sent, o, t in zip(texts, outputs, targets):
        a = [k for k in clean_split(o.lower()) if k != "()"]
        g = clean_split(t.lower())
        attempts += len(a)
        total_gold += len(g)

        # use exact match
        correct += len([k for k in a if k in g])

    if 0 in [total_gold, attempts, correct]:
        printer("No accurate extractions have been made. Continuing training...\n")
        return 0

    printer(
        f"Out of {total_gold} NEs, the model extracted {correct} triples or entities correctly in {attempts} attempts.\n")
    p = correct / attempts
    r = correct / total_gold
    f1 = 2 * p * r / (p + r)
    printer(f"Precision: {p}, recall: {r}, F1 score: {f1}\n")

    for i in range(3):
        printer("Input text:    %s\n" % texts[i])
        printer("Actual NEs:    %s\n" % targets[i])
        printer("Predicted NEs: %s\n" % outputs[i])
        printer("=====================================================================\n")

    return f1


def multitask_score(texts, outputs, targets, printer=print):
    # assume NER-OIE multitask model
    ner_outputs = []
    ner_targets = []
    oie_outputs = []
    oie_targets = []

    for output, target in zip(outputs, targets):
        ner_os = []
        oie_os = []
        ner_ts = []
        oie_ts = []
        output = output.replace('() ', '')
        target = target.replace('() ', '')

        for o in clean_split(output):
            if any([k in o for k in ['LOC: ', 'PER: ', 'ORG: ', 'PRO: ']]):
                ner_os.append(f"({o})")
            else:
                oie_os.append(f"({o})")
        for t in clean_split(target):
            if any([k in t for k in ['LOC: ', 'PER: ', 'ORG: ', 'PRO: ']]):
                ner_ts.append(f"({t})")
            else:
                oie_ts.append(f"({t})")
        ner_outputs.append(' '.join(ner_os))
        ner_targets.append(' '.join(ner_ts))
        oie_outputs.append(' '.join(oie_os))
        oie_targets.append(' '.join(oie_ts))

    object_generation_score(texts, ner_outputs, ner_targets)
    object_generation_score(texts, oie_outputs, oie_targets)


def triple_soft_match_score(texts: List[str],
                            output_list: List[List[Dict]],
                            target_list: List[List[Dict]],
                            outfile=None,
                            printer=print):
    """
    Calculates various OIE metrics including Handcrafted_gold scores, exact matche F1.

    :param outfile:
    :param texts: List of test set sentences
    :param output_list: List of output dictionaries in the form of {"head": str, "predicate": str, "tail": str}
    :param target_list: List of target (gold) dictionaries in the same form as output_list
    :param printer: Printing function--can be a logger or print, or other custom function

    :return: None
    """

    def evaluate(sent, gold_head, gold_pred, gold_tail, extractions):
        '''
        Evaluates predicate candidates and selects best one for given gold triple

        Args:
            sent: Original CaRB sentence in string format
            gold_head: Gold or original CaRB head in string format
            gold_pred: Gold or original CaRB predicate in string format
            gold_tail: Gold or original CaRB tail in string format
            extractions: List of extracted triples
            [
                {
                    "head": "This",
                    "predicate": "is",
                    "tail": "an example sentence"
                },
                ...
            ]

        Returns:
            Tuple of strings in the form of (judgement, best predicate)
        '''
        candidates = {}
        for extraction in extractions:
            extracted_pred = extraction['predicate']
            if extraction is None:
                continue
            elif is_same(gold_pred, extracted_pred):
                return "good", extraction
            elif overlap_rate(gold_pred, extraction['predicate']) < 0.25 or extracted_pred == '-':
                continue
            elif not is_sequential(sent, gold_head, gold_pred, gold_tail, extracted_pred):
                # if nonseq
                if diff_is_adv(gold_pred, extracted_pred):
                    # if diff is adv then acceptable
                    if "acceptable" in candidates:
                        candidates["acceptable"].append(extraction)
                    else:
                        candidates["acceptable"] = [extraction]
                else:
                    # else incorrect
                    if "incorrect" in candidates:
                        candidates["incorrect"].append(extraction)
                    else:
                        candidates["incorrect"] = [extraction]
            elif overlap_rate(gold_head, extracted_pred) > 0.75 or \
                    overlap_rate(gold_tail, extracted_pred) > 0.75:
                # also incorrect if trespass head / tail boundary too much
                if "acceptable" in candidates:
                    candidates["acceptable"].append(extraction)
                else:
                    candidates["acceptable"] = [extraction]
            else:  # no issues then acceptable, I guess
                if "acceptable" in candidates:
                    candidates["acceptable"].append(extraction)
                else:
                    candidates["acceptable"] = [extraction]

        # loop over candidates and their judgements
        if "acceptable" in candidates:
            return "acceptable", shortest_extraction(candidates["acceptable"])
        elif "incorrect" in candidates:
            return "incorrect", shortest_extraction(candidates["incorrect"])
        else:
            return "no detection", {"head": "-", "predicate": "-", "tail": "-"}

    def diff_is_adv(gold_pred, extraction):
        '''
        Determines if the extraction is only missing a single adverb from gold predicate.

        Args:
            gold_pred
            extraction

        Returns:
            A boolean argument
        '''
        advs = ["afterward", "already", "almost", "back", "better", "best", "even", "far", "fast", "hard",
                "here", "how", "late", "long", "low", "more", "near", "never", "next", "now", "often",
                "quick", "rather", "slow", "so", "soon", "still", "then", "today", "tomorrow", "too", "very", "well",
                "where", "yesterday"]
        if overlap_rate(extraction, gold_pred) != 1:
            # extraction not entirely in gold_pred
            return False
        diff = []
        ext_words = extraction.split(' ')
        for gold_word in gold_pred.split(' '):
            if gold_word not in ext_words:
                diff.append(gold_word)
        if len(diff) != 1:
            return False
        diff = diff[0]
        if diff[-2:] == 'ly' or diff in advs:
            return True
        else:
            return False

    def is_same(gold_pred, extraction):
        '''
        Whether gold predicate and extracted predicate are equal
        '''
        if gold_pred.replace(' ', '').lower() == extraction.replace(' ', '').lower():
            return True
        else:
            return False

    def overlap_rate(this, that):
        '''
        How much of this is also in that?

        Args:
            this: denominator
            that: numerator

        Returns:
            In decimal form how much of this is also in that
        '''
        this_words = this.split(' ')
        this_len = len(this_words)
        that_words = that.split(' ')
        overlap = 0
        for this_word in this_words:
            if this_word in that_words:
                overlap += 1
        return overlap / this_len

    def shortest_extraction(extractions: List[Dict]) -> Dict:
        # use dummy dictionary
        shortest = {"predicate": ""}
        shortest_len = 0

        # loop over extraction dictionaries
        for extraction in extractions:
            extracted_pred = extraction['predicate']
            ext_len = len(extracted_pred.split(' '))
            if shortest['predicate'] == '' or ext_len < shortest_len:
                shortest = extraction
                shortest_len = ext_len
            else:
                continue
        return shortest

    def is_sequential(sentence, gold_head, gold_pred, gold_tail, extraction):
        ext_nosp = extraction.replace(' ', '').lower()
        if ext_nosp in sentence.replace(' ', '').lower():
            return True
        elif ext_nosp in (gold_head + gold_pred + gold_tail).replace(' ', '').lower():
            return True
        else:
            return False

    # compare output and targets
    head_exact_match = 0
    tail_exact_match = 0
    pred_exact_match = 0
    triple_exact_match = 0
    incorrect_inferences = []
    num_attempts = sum([len(k) for k in output_list])
    num_gold = sum([len(k) for k in target_list])
    printer(f"Extracted {num_attempts} extractions from {len(texts)} sentences. Gold set has "
            f"{num_gold} extractions.\n")
    if outfile is not None:
        f = open(outfile, 'w')
        f.write('\t'.join(["sentence", "head", "pred", "tail"]) + "\n")
    judgements = {"good": 0, "acceptable": 0, "incorrect": 0, "no detection": 0}
    for sentence, sentence_outputs, sentence_targets in zip(texts, output_list, target_list):

        # loop over target and outputs
        for t in sentence_targets:
            gold_head = t['head']
            gold_pred = t['predicate']
            gold_tail = t['tail']
            judgement, silver = evaluate(sentence, gold_head, gold_pred, gold_tail, sentence_outputs)
            judgements[judgement] += 1

            # record extractions
            if outfile is not None:
                f.write('\t'.join([sentence, gold_head, gold_pred, gold_tail, judgement,
                                   silver["head"], silver["predicate"], silver["tail"]]) + "\n")

            # count exact matches
            triplewise_match = 0
            if gold_head in [s['head'] for s in sentence_outputs]:
                head_exact_match += 1
                triplewise_match += 1
            if gold_pred in [s['predicate'] for s in sentence_outputs]:
                pred_exact_match += 1
                triplewise_match += 1
            if gold_tail in [s['tail'] for s in sentence_outputs]:
                tail_exact_match += 1
                triplewise_match += 1
            if triplewise_match == 3:
                triple_exact_match += 1

    # calculate metrics
    score = (judgements['good'] + 0.5 * judgements['acceptable'] - judgements['incorrect']) / sum(judgements.values())
    printer(f"Model finds {judgements['good']} good extractions, {judgements['acceptable']} acceptable extractions,"
            f" {judgements['incorrect']} incorrect extractions, and {judgements['no detection']} no detections "
            f"to achieve a score of {score}.'\n")
    printer(f"Number of exactly matched heads is {head_exact_match},"
            f" predicates {pred_exact_match}, tails {tail_exact_match}, and triples {triple_exact_match}\n")

    if 0 in [num_attempts, num_gold]:
        print("No good extractions have been made. Continuing training...")
        return 0
    else:
        ps = [k / num_attempts for k in [head_exact_match, pred_exact_match, tail_exact_match, triple_exact_match]]
        rs = [k / num_gold for k in [head_exact_match, pred_exact_match, tail_exact_match, triple_exact_match]]
        f1 = [2 * p * r / (p + r) for p, r in zip(ps, rs)]
        printer(f"|    |  P  |  R  | F1  |\n"
                f"------------------------\n"
                f"|Head|{'%.3f' % (ps[0])}|{'%.3f' % (rs[0])}|{'%.3f' % (f1[0])}|\n"
                f"|Pred|{'%.3f' % (ps[1])}|{'%.3f' % (rs[1])}|{'%.3f' % (f1[1])}|\n"
                f"|Tail|{'%.3f' % (ps[2])}|{'%.3f' % (rs[2])}|{'%.3f' % (f1[2])}|\n"
                f"| Tot|{'%.3f' % (ps[3])}|{'%.3f' % (rs[3])}|{'%.3f' % (f1[3])}|\n")

        # show examples
        for i in incorrect_inferences:
            printer(f"sentence: {i[0][35:]}")
            printer(f"gold extractions: {i[1]}")
            printer(f"predicted extractions: {i[2]}\n\n")

        return f1[1]  # return predicate f1


def text2carb(ckpt: Union[str, None], model, input_dataset, tokenizer, max_len, batch_size,
              skip_special_tokens=True, num_beams=4,
              printer="outfile", seed=0):
    if ckpt:
        model = model.load_from_checkpoint(ckpt)
    else:
        pass

    if printer == "outfile":
        ckpt_name = sorted(ckpt.split('/'), key=lambda k: len(k), reverse=True)[0]
        printer = open(f"/data/carb/{ckpt_name}_{seed}.oie", "w").write
    elif type(printer) == str:
        printer = open(printer, 'w').write
    else:
        printer = print

    dataloader = DataLoader(input_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    model.model.eval()
    model = model.to("cuda")
    outputs = []

    for batch in tqdm(dataloader):
        if 'input_ids' in batch:
            outs = model.model.generate(input_ids=batch['input_ids'].to("cuda"),
                                        attention_mask=batch['attention_mask'].to("cuda"),
                                        max_length=max_len, num_beams=num_beams
                                        )
        else:
            outs = model.model.generate(input_ids=batch['source_ids'].to("cuda"),
                                        attention_mask=batch['source_mask'].to("cuda"),
                                        max_length=max_len, num_beams=num_beams
                                        )
        dec = [
            tokenizer.decode(ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False).strip()
            for ids in
            outs]
        outputs.extend(dec)

    structured_outputs, _ = text2triple(outputs, [""] * len(outputs))

    for t, os in zip(input_dataset.data, structured_outputs):
        printer(t + "\n")
        for o in os:
            printer(f"1.00: ({o['head']}; {o['predicate']}; {o['tail']})\n")
        printer('\n')


def mpm_score(texts, outputs, targets, printer=print):
    printer(f"{len(texts)} text chunks \n")

    # calculate metric
    hit = 0
    false_alarm = 0
    miss = 0
    foul = 0
    correct_rejection = 0

    problem = 0
    attempt = 0

    len_mismatch = 0
    for source, output, target in zip(texts, outputs, targets):
        s_tokens = source[32:].split(' ')  # prompt is 32 chars long
        o_tokens = output.split(' ')
        t_tokens = target.split(' ')

        if len(s_tokens) != len(o_tokens) or len(s_tokens) != len(t_tokens):
            len_mismatch += 1
            printer(
                f"Found length mismatch between source {len(s_tokens)}, output {len(o_tokens)}, target {len(t_tokens)}\n")
            printer("Skipping...")
            min_len = min([len(s_tokens), len(o_tokens), len(t_tokens)])
            s_tokens = s_tokens[:min_len]
            o_tokens = o_tokens[:min_len]
            t_tokens = t_tokens[:min_len]
            continue

        for s, o, t in zip(s_tokens, o_tokens, t_tokens):
            # is problem
            if s != t:
                problem += 1
                if s != o:
                    attempt += 1
                    if o == t:
                        hit += 1
                    elif o != t:
                        foul += 1
                elif s == o:
                    miss += 1

            # not problem
            elif s == t:
                if s != o:
                    attempt += 1
                    false_alarm += 1
                elif s == o:
                    correct_rejection += 1

        p = hit / attempt
        r = hit / problem

    if 0 in [attempt, hit, problem]:
        print("No good extractions have been made. Continuing training...")
    else:

        printer(f"From a corpus of {problem} problems, {attempt} attempts were made. \n"
                f"Among them, we have {hit} hits, {correct_rejection} correct rejections, and {foul} fouls, "
                f"{miss} misses, {false_alarm} false alarms. \n"
                f"Precision: {p}, recall: {r}, f1: {2 * p * r / (p + r)} \n"
                f"We also report {len_mismatch} length mismatches."
                )

        # show sample
        for i in range(3):
            printer("Input sentence:     %s" % texts[i])
            printer("Actual sentence:    %s" % targets[i])
            printer("Predicted sentence: %s" % outputs[i])
            printer("=====================================================================\n")

    return 2 * p * r / (p + r)


def sbd_score(texts, outputs, targets, printer=print):
    # use macro accuracy for each tag in ["B", "I", "E"]

    gold_b, gold_i, gold_e = 0, 0, 0
    correct_b, correct_i, correct_e = 0, 0, 0
    attempt_b, attempt_i, attempt_e = 0, 0, 0

    for o, t in zip(outputs, targets):
        o = o.replace(" <EOS> ", "<EOS>")
        t = t.replace(" <EOS> ", "<EOS>")
        t_tokens = []
        for t_token in t.split(' '):
            if '<EOS>' in t_token:

                # single-word sentence take the E tag
                t_subtokens = t_token.split('<EOS>')

                gold_e += (len(t_subtokens) - 1)
                t_tokens.extend([(k, "E") for k in t_subtokens[:-1]])

                t_tokens.append((t_subtokens[-1], "B"))
                gold_b += 1


            else:
                t_tokens.append((t_token, "I"))
                gold_i += 1
        for o_token in o.split(' '):
            if '<EOS>' in o_token:
                o_subtokens = o_token.split('<EOS>')

                # single-word sentences take the E tag
                for o_subtoken in o_subtokens[:-1]:
                    attempt_e += 1
                    if (o_subtoken, "E") in t_tokens:
                        correct_e += 1

                attempt_b += 1
                if (o_subtokens[-1], "B") in t_tokens:
                    correct_b += 1

            else:
                if (o_token, "I") in t_tokens:
                    correct_i += 1
                attempt_i += 1

    # show sample
    for i in range(3):
        printer("Input sentence:     %s\n" % texts[i])
        printer("Actual sentence:    %s\n" % targets[i])
        printer("Predicted sentence: %s\n" % outputs[i])
        printer("=====================================================================\n")

    if 0 in [attempt_i, attempt_b, attempt_e]:
        print([attempt_i, attempt_b, attempt_e])
        return 0
    else:
        b_p, b_r = correct_b / attempt_b, correct_b / gold_b
        i_p, i_r = correct_i / attempt_i, correct_i / gold_i
        e_p, e_r = correct_e / attempt_e, correct_e / gold_e

        b_f1 = b_p * b_r * 2 / (b_p + b_r)
        i_f1 = i_p * i_r * 2 / (i_p + i_r)
        e_f1 = e_p * e_r * 2 / (e_p + e_r)

        macro_p = (b_p + i_p + e_p) / 3
        macro_r = (b_r + i_r + e_r) / 3
        macro_f1 = (b_f1 + i_f1 + e_f1) / 3

        printer(f"|    |  P  |  R  | F1  |\n"
                f"------------------------\n"
                f"|Begi|{'%.3f' % b_p}|{'%.3f' % b_r}|{'%.3f' % b_f1}|\n"
                f"|  In|{'%.3f' % i_p}|{'%.3f' % i_r}|{'%.3f' % i_f1}|\n"
                f"| End|{'%.3f' % e_p}|{'%.3f' % e_r}|{'%.3f' % e_f1}|\n"
                f"|Macr|{'%.3f' % macro_p}|{'%.3f' % macro_r}|{'%.3f' % macro_f1}|\n")
        return macro_f1
