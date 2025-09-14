from utils import logger
from datasets import load_dataset
import json

# call: python -m data.sst2
logger = logger()
#sentiment on movie reviews
def generate_sst2():
    sst2_train = load_dataset("glue", "sst2", split="train")
    sst2_dev = load_dataset("glue", "sst2", split="validation")
    sst2_test = load_dataset("glue", "sst2", split="test")
    splits = ["sst2_train", "sst2_dev", "sst2_test"]
    myMap = {"sst2_train": sst2_train, "sst2_dev": sst2_dev, "sst2_test": sst2_test}
    myDict = {0: "negative", 1: "positive"}
    for outputPath in splits:
        excerpt_count = 0
        print("preparing", outputPath)
        with open(f"outputs/datasets/{outputPath}.jsonl", 'w', encoding='utf-8') as fout:
            for example in myMap[outputPath]:
                # Note that the test set does not have labels and it's all -1
                text = example["sentence"]
                label = example["label"]
                target = myDict[label] if label in myDict else "missing"
                json.dump({'source': text, 'target': target}, fout, ensure_ascii=False)
                fout.write('\n')

                excerpt_count += 1
        print(f"prepared from {excerpt_count} excerpts for {outputPath}")

if __name__ == "__main__":
    print("sst2 dataset \n")
    generate_sst2()