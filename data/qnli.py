from utils import logger
from datasets import load_dataset
import json

# call: python -m data.qnli
logger = logger()

# 1 for entailment, 0 for not entailment
def generate_qnli():
    qnli_train = load_dataset("glue", "qnli", split="train")
    qnli_dev = load_dataset("glue", "qnli", split="validation")
    qnli_test = load_dataset("glue", "qnli", split="test")
    splits = ["qnli_train", "qnli_dev", "qnli_test"]
    myMap = {"qnli_train": qnli_train, "qnli_dev": qnli_dev, "qnli_test": qnli_test}
    for outputPath in splits:
        excerpt_count = 0
        print("preparing", outputPath)
        with open(f"outputs/datasets/{outputPath}.jsonl", 'w', encoding='utf-8') as fout:
            for example in myMap[outputPath]:
                # Note that the test set does not have labels and it's all -1
                text1 = example["question"]
                text2 = example["sentence"]
                label = example["label"]
                json.dump({'source': "(" + text1 + ") (" + text2 + ")", 'target': str(label)}, fout, ensure_ascii=False)
                fout.write('\n')

                excerpt_count += 1
        print(f"prepared from {excerpt_count} excerpts for {outputPath}")

if __name__ == "__main__":
    print("qnli dataset \n")
    generate_qnli()
