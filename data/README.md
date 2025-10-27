Directory for datasets. 
`modules.py` provides classes with some helper functions one can inherit to adapt to new datasets. 

### Antilles

[ANTILLES](https://github/qanastek/ANTILLES) 
is a POS-tagging corpus, 
originally created in 2015 
based on the universal dependency treebank v2.0.

_Caveats:_ 
`.conllu` files are needed from the source repository. 
Usage:

```
python -m data.antilles <repo_dir> 
```

### CaRB

[CaRB](https://github.com/dair-iitd/CaRB)
is a crowdsourced benchmark dataset for OIE. 

_Caveats:_
`.tsv` files are needed from the source repository.
Usage:

```
python -m data.carb <repo_dir> 
```

### CoNLL 2000

CoNLL 2000 is a Chunking shared-task. 
We use the hugging face repo [here](https://huggingface.co/datasets/haeunkim/spacy-conll2000-pos).
[Another repo](https://huggingface.co/datasets/eriktks/conll2000) seems to not be working. 
Cite: 
> Erik F. Tjong Kim Sang and Sabine Buchholz. 2000. 
> [Introduction to the CoNLL-2000 Shared Task Chunking.](https://aclanthology.org/W00-0726/) 
> In Fourth Conference on Computational Natural Language Learning and the Second Learning Language in Logic Workshop.

### CoNLL 2003

CoNLL 2003 is a language-independent NER shared-task. 
We use the hugging face repo
[here](https://huggingface.co/datasets/lhoestq/conll2003). 
Cite:
> Erik F. Tjong Kim Sang and Fien De Meulder. 2003. 
> [Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition.](https://aclanthology.org/W03-0419/) 
> In Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003, pages 142–147.

### CoNLL 2004

CoNLL 2004 is a benchmark dataset used for RE. 
We use the hugging face repo 
[here](https://huggingface.co/datasets/DFKI-SLT/conll04). 
Cite: 
> Dan Roth and Wen-tau Yih. 2004. 
> [A Linear Programming Formulation for Global Inference in Natural Language Tasks.](https://aclanthology.org/W04-2401)
> In Proceedings of the Eighth Conference on Computational Natural Language Learning (CoNLL-2004) at HLT-NAACL 2004, pages 1–8, Boston, Massachusetts, USA. Association for Computational Linguistics.

### CoNLL 2012

OntoNotes v5.0 is the final version of OntoNotes corpus, 
a large-scale, multi-genre, multilingual corpus manually annotated with syntactic, semantic and discourse information.
This dataset is the version of OntoNotes v5.0 extended and is used in the CoNLL 2012 shared task. 
We use the hugging face dataset
[here](https://huggingface.co/datasets/ontonotes/conll2012_ontonotesv5). 
Cite:
> Sameer Pradhan, Alessandro Moschitti, Nianwen Xue, Hwee Tou Ng, Anders Björkelund, Olga Uryupina, Yuchen Zhang, and Zhi Zhong. 2013. 
> [Towards Robust Linguistic Analysis using OntoNotes.](https://aclanthology.org/W13-3516/)
> In Proceedings of the Seventeenth Conference on Computational Natural Language Learning, pages 143–152, Sofia, Bulgaria. Association for Computational Linguistics.

### Genia

A processed version of the GENIA corpus —
a semantically annotated corpus for bio-textmining, 
specialized for NER.
We use the hugging face dataset
[here](https://huggingface.co/datasets/chufangao/GENIA-NER). 
Cite:
> J.-D. Kim, T. Ohta, Y. Tateisi, J. Tsujii, 
> GENIA corpus—a semantically annotated corpus for bio-textmining, 
> Bioinformatics, Volume 19, Issue suppl_1, July 2003, Pages i180–i182, https://doi.org/10.1093/bioinformatics/btg1023

### MultiNERD

A multilingual NER dataset. 
We use the hugging face dataset
[here](https://huggingface.co/datasets/tner/multinerd). 
Cite:
> Simone Tedeschi and Roberto Navigli. 2022. 
> [MultiNERD: A Multilingual, Multi-Genre and Fine-Grained Dataset for Named Entity Recognition (and Disambiguation).](https://aclanthology.org/2022.findings-naacl.60/)
> In Findings of the Association for Computational Linguistics: NAACL 2022, pages 801–812, Seattle, United States. Association for Computational Linguistics.

### OIE 2016

A benchmark dataset for OIE. 
Cite:
> Gabriel Stanovsky and Ido Dagan. 2016. 
> [Creating a Large Benchmark for Open Information Extraction.](https://aclanthology.org/D16-1252/)
> In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2300–2305, Austin, Texas. Association for Computational Linguistics.

_Caveats:_
`.oie` files are needed from the source repository.
A script need to be run first to generate an aggregated file. 
Usage:

```
cd <oie2016-dir>
./create_oie_corpus.sh
cd <pr-dir>
python -m data.oie_2016 <oie2016-dir> 
```

### Ontonotes 5

An older version of [CoNLL 2012](#conll-2012). 
We use the hugging face dataset
[here](https://huggingface.co/datasets/tner/ontonotes5). 

> Eduard Hovy, Mitchell Marcus, Martha Palmer, Lance Ramshaw, and Ralph Weischedel. 2006. 
> [OntoNotes: The 90% Solution.](https://aclanthology.org/N06-2015/)
> In Proceedings of the Human Language Technology Conference of the NAACL, Companion Volume: Short Papers, pages 57–60, New York City, USA. Association for Computational Linguistics.

### ParTUT

[UD_French-ParTUT](https://github.com/UniversalDependencies/UD_French-ParTUT)
is a conversion of a multilingual parallel treebank developed at the University of Turin, 
and consisting of a variety of text genres, 
including talks, legal texts and Wikipedia articles, among others.
We use the hugging face dataset 
[here](https://huggingface.co/datasets/CATIE-AQ/universal_dependencies_fr_partut_fr_prompt_pos). 
Cite:
> Manuela Sanguinetti, Cristina Bosco. 2014. PartTUT: The Turin University Parallel Treebank. In Basili, Bosco, Delmonte, Moschitti, Simi (editors) Harmonization and development of resources and tools for Italian Natural Language Processing within the PARLI project, LNCS, Springer Verlag

> Manuela Sanguinetti, Cristina Bosco. 2014. Converting the parallel treebank ParTUT in Universal Stanford Dependencies. In Proceedings of the 1rst Conference for Italian Computational Linguistics (CLiC-it 2014), Pisa (Italy)

> Cristina Bosco, Manuela Sanguinetti. 2014. Towards a Universal Stanford Dependencies parallel treebank. In Proceedings of the 13th Workshop on Treebanks and Linguistic Theories (TLT-13), Tubingen (Germany)
 
### WikiAnn

We use the hugging face dataset
[here](https://huggingface.co/datasets/tner/wikiann). 
Cite:
> Afshin Rahimi, Yuan Li, Trevor Cohn. 2019.
> Massively Multilingual Transfer for NER. 
> [arXiv 1902.00193](https://arxiv.org/abs/1902.00193)