# punctuation-restoration-for-structure-understanding

This is our code that accompanies a RepL4NLP Paper 
[Punctuation restoration improves structure understanding without supervision](https://aclanthology.org/2025.repl4nlp-1.10/), 
where we apply punctuation restoration as an unsupervised structure learning objective 
that complements masked language modeling in T5 models. 
The additional training in punctuation restoration results in
improved in-distribution and out-of-distribution performance in various structure-related NLP tasks
such as named entity recognition, open information extraction, and semantic role labeling.

**Important**: Make sure you are at the repo root before running python files!

### Data-prepping

Run a file under `data/` to generate training data using the specific dataset. 
Shared-task datasets' files will generate a separate JSONL file for each implemented task. 
JSONL files are under `outputs/datasets/`. 

For example, the following loads the CoNLL 2003 dataset and generates data for POS and NER. 

```commandline
python -m data.conll_2003
```

for now, it is necessary to use the `-m` and run it as a module. 

OIE datasets may require additional local data files. Refer to implementation details. 

### Pre-training and Fine-tuning

```commandline
python train.py TASK -n CKPT_NAME
```

where `TASK` is one of

- `chunking`: Chunking
- `mlm`: Masked Language Modelling
- `ner`: Named Entity Recognition
- `oie`: Open Information Extraction
- `pos`: Part-of-speech Tagging
- `pr`: Punctuation Restoration
- `re`: Relation Extraction
- `srl`: Semantic Role Labelling

and `CKPT_NAME` is a string that will be prefixed to the generating checkpoint. 
Checkpoints are under `outputs/checkpoints/`. 

Optional arguments: 

- `-d`: path to a JSONL file containing training data. By default, a file associated with the task is used. 
- `-e`: number of epochs to run. `-e 30` to run exactly 30 epochs, `-e 1-3` to run at least 1 and at most 3.
- `-k`: number of epochs to save. This saves the epochs with the $k$-most minimum validation losses.
- `-r`: path to a checkpoint to resume training on.
- `-s`: index of an epoch to save. **Starts at 0 (0 is first epoch)**. Can be provided multiple times.
- `--learning-rate`: learning rate. 
- `--max-seq-len`: max token length in a sequence. 
- `--precision`: precision for Lightning trainer. Could be one of
  - `64`, `64-true`: 64-bit
  - `32`, `32-true`: 32-bit
  - `16`, `16-mixed`: 16-bit --- 5-bit exponent, 10-bit fraction
  - `bf16`, `bf16-mixed`: 16-bit --- 8-bit exponent, 7-bit fraction
- `--save-last-epoch`: save the last epoch. 
- `--seed`: seed for random generation. 

### Evaluating

```commandline
python eval.py TASK -c CKPT -n MODEL_NAME
```

where `TASK` is one of

- `chunking`: Chunking
- `ner`: Named Entity Recognition
- `oie`: Open Information Extraction
- `pos`: Part-of-speech Tagging
- `pr`: Punctuation Restoration
- `re`: Relation Extraction
- `sbd`: Sentence Boundary Detection
- `srl`: Semantic Role Labelling

`CKPT` is the path to the checkpoint used for generating outputs, and `MODEL_NAME` is a string that will be used for the generation result cache and printed information. 
Generation result cache files are under `outputs/generated/`. 

Optional arguments: 

- `-d`: path to a JSONL file containing evaluation data. By default, a file associated with the task is used. 
- `--max-seq-len`: max token length in a sequence. 
- `--strict`: use a stricter evaluation metric dependent on the task. Might not always have an effect. 

### Logs

Tensorboard logs can be viewed at [localhost:6006](localhost:6006) by:
```commandline
tensorboard --logdir=outputs/logs --host=0.0.0.0
```

If you're on GU CLI, refer to the [GU CLI remote dev guide](https://github.com/Aatlantise/gu-cli-remote-dev)
to set up a tunnel to view the logs on your local machine.



