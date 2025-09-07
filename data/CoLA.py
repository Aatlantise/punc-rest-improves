from utils import logger
from datasets import load_dataset
import json

# call: python -m data.CoLA
logger = logger()

def generate_cola():
    cola_train = load_dataset("glue", "cola", split="train")
    cola_dev = load_dataset("glue", "cola", split="validation")
    cola_test = load_dataset("glue", "cola", split="test")
    splits = ["cola_train", "cola_dev", "cola_test"]
    myMap = {"cola_train": cola_train, "cola_dev": cola_dev, "cola_test": cola_test}
    labelMap = {}
    for outputPath in splits:
        excerpt_count = 0
        print("preparing", outputPath)
        with open(f"outputs/datasets/{outputPath}.jsonl", 'w', encoding='utf-8') as fout:
            for example in myMap[outputPath]:
                # Note that the test set does not have labels and it's all -1
                text = example["sentence"]
                label = example["label"]
                json.dump({'source': text, 'target': str(label)}, fout, ensure_ascii=False)
                fout.write('\n')

                excerpt_count += 1
        print(f"prepared from {excerpt_count} excerpts for {outputPath}")

if __name__ == "__main__":
    print("cola dataset \n")
    generate_cola()