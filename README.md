# punctuation-restoration-for-structure-understanding

This is our code that accompanies a RepL4NLP Paper 
[Punctuation restoration improves structure understanding without supervision](https://aclanthology.org/2025.repl4nlp-1.10/), 
where we apply punctuation restoration as an unsupervised structure learning objective 
that complements masked language modeling in T5 models. 
The additional training in punctuation restoration results in
improved in-distribution and out-of-distribution performance in various structure-related NLP tasks
such as named entity recognition, open information extraction, and semantic role labeling.

### PR training data
The authors of the paper use an unreleased internal source to compile a PR training corpus.
We opt to use Wikipedia to implement PR-T5.

The following script produces a `punctuation_restoration_dataset.jsonl`
that contains 450k examples of 150 words, similarly to the paper:

```angular2html
python prep-pr-data.py
```

### Additional training on PR
Then, the following script performs training

```angular2html
python main.py
```

Tensorboard logs can be viewed at [localhost:6006](localhost:6006) by:
```angular2html
tensorboard --logdir=outputs/logs/version_0 --host=0.0.0.0
```

If you're on GU CLI, refer to the [GU CLI remote dev guide](https://github.com/Aatlantise/gu-cli-remote-dev)
to set up a tunnel to view the logs on your local machine.



