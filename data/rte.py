from utils import logger
from datasets import load_dataset
import json

# call: python -m data.rte
logger = logger()

# 1 for entailment, 0 for not entailment
def generate_rte():
    rte_train = load_dataset("glue", "rte", split="train")
    rte_dev = load_dataset("glue", "rte", split="validation")
    rte_test = load_dataset("glue", "rte", split="test")
    splits = ["rte_train", "rte_dev", "rte_test"]
    myMap = {"rte_train": rte_train, "rte_dev": rte_dev, "rte_test": rte_test}
    for outputPath in splits:
        excerpt_count = 0
        print("preparing", outputPath)
        with open(f"outputs/datasets/{outputPath}.jsonl", 'w', encoding='utf-8') as fout:
            for example in myMap[outputPath]:
                # Note that the test set does not have labels and it's all -1
                text1 = example["sentence1"]
                text2 = example["sentence2"]
                label = example["label"]
                json.dump({'source': "(" + text1 + ") (" + text2 + ")", 'target': str(label)}, fout, ensure_ascii=False)
                fout.write('\n')

                excerpt_count += 1
        print(f"prepared from {excerpt_count} excerpts for {outputPath}")

if __name__ == "__main__":
    print("rte dataset \n")
    generate_rte()
