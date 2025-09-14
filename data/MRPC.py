from utils import logger
from datasets import load_dataset
import json

# call: python -m data.mrpc
logger = logger()

# paraphrase classification task (semantically equal or not)
def generate_mrpc():
    mrpc_train = load_dataset("glue", "mrpc", split="train")
    mrpc_dev = load_dataset("glue", "mrpc", split="validation")
    mrpc_test = load_dataset("glue", "mrpc", split="test")
    splits = ["mrpc_train", "mrpc_dev", "mrpc_test"]
    myMap = {"mrpc_train": mrpc_train, "mrpc_dev": mrpc_dev, "mrpc_test": mrpc_test}
    myDict = {0: "not_equivalent", 1: "equivalent"}
    for outputPath in splits:
        excerpt_count = 0
        print("preparing", outputPath)
        with open(f"outputs/datasets/{outputPath}.jsonl", 'w', encoding='utf-8') as fout:
            for example in myMap[outputPath]:
                # Note that the test set does not have labels and it's all -1
                text1 = example["sentence1"]
                text2 = example["sentence2"]
                label = example["label"]
                target = myDict[label] if label in myDict else "missing"
                json.dump({'source': "(" + text1 + ") (" + text2 + ")", 'target': target}, fout, ensure_ascii=False)
                fout.write('\n')

                excerpt_count += 1
        print(f"prepared from {excerpt_count} excerpts for {outputPath}")

if __name__ == "__main__":
    print("mrpc dataset \n")
    generate_mrpc()
