from utils import logger
from datasets import load_dataset
import json

# call: python -m data.stsb
logger = logger()

# similarity scores which is a numeric value
def generate_stsb():
    stsb_train = load_dataset("glue", "stsb", split="train")
    stsb_dev = load_dataset("glue", "stsb", split="validation")
    stsb_test = load_dataset("glue", "stsb", split="test")
    splits = ["stsb_train", "stsb_dev", "stsb_test"]
    myMap = {"stsb_train": stsb_train, "stsb_dev": stsb_dev, "stsb_test": stsb_test}
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
    print("stsb dataset \n")
    generate_stsb()
