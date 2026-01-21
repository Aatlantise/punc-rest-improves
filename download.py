from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from argparse import ArgumentParser

DATASET_NAME = "ashraq/financial-news-articles"

def download():
    print("Downloading financial news data...")
    ds = load_dataset("ashraq/financial-news-articles", split='train')
    ds.save_to_disk("./data/finance-news")

    print("Downloading T5 tokenizer and model...")
    T5ForConditionalGeneration.from_pretrained('google-t5/t5-base').save_pretrained("./t5-base")
    T5Tokenizer.from_pretrained('google-t5/t5-base').save_pretrained("./t5-base")

if __name__ == "__main__":
    download()