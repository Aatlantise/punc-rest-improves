from utils import logger
from datasets import load_dataset
import json

# call: python -m data.mnli
logger = logger()

# inference task (entailment, neutral, contradiction)
def generate_mnli():
    mnli_train = load_dataset("glue", "mnli", split="train")
    mnli_dev_matched = load_dataset("glue", "mnli", split="validation_matched")
    mnli_dev_mismatched = load_dataset("glue", "mnli", split="validation_mismatched")
    mnli_test_matched = load_dataset("glue", "mnli", split="test_matched")
    mnli_test_mismatched = load_dataset("glue", "mnli", split="test_mismatched")
    splits = ["mnli_train", "mnli_dev_matched", "mnli_dev_mismatched", "mnli_test_matched", "mnli_test_mismatched"]
    myMap = {"mnli_train": mnli_train, "mnli_dev_matched": mnli_dev_matched, "mnli_dev_mismatched": mnli_dev_mismatched, "mnli_test_matched": mnli_test_matched, "mnli_test_mismatched": mnli_test_mismatched}
    myDict = {0: "entailment", 1: "neutral", 2: "contradiction"}
    for outputPath in splits:
        excerpt_count = 0
        print("preparing", outputPath)
        with open(f"outputs/datasets/{outputPath}.jsonl", 'w', encoding='utf-8') as fout:
            for example in myMap[outputPath]:
                # Note that the test set does not have labels and it's all -1
                premise = example["premise"]
                hypothesis = example["hypothesis"]
                label = example["label"]
                target = myDict[label] if label in myDict else "missing"
                json.dump({'source': "(" + premise + ") (" + hypothesis + ")", 'target': target}, fout, ensure_ascii=False)
                fout.write('\n')

                excerpt_count += 1
        print(f"prepared from {excerpt_count} excerpts for {outputPath}")

if __name__ == "__main__":
    print("mnli dataset \n")
    generate_mnli()

