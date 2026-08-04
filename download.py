from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModel, AutoModelForCausalLM, AutoTokenizer
from argparse import ArgumentParser

DATASET_NAME = "ashraq/financial-news-articles"

def download():
    print("Downloading financial news data...")
    ds = load_dataset("ashraq/financial-news-articles", split='train')
    ds.save_to_disk("./data/finance-news")

    print("Downloading T5 tokenizer and model...")
    T5ForConditionalGeneration.from_pretrained('google-t5/t5-base').save_pretrained("./t5-base")
    T5Tokenizer.from_pretrained('google-t5/t5-base').save_pretrained("./t5-base")
    
    print("Downloading bert tokenizer and model...")
    AutoModel.from_pretrained("google-bert/bert-base-uncased").save_pretrained("./bert-base-uncased")
    AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", use_fast=True).save_pretrained("./bert-base-uncased")
    
    print("Downloading gpt2 medium tokenizer and model...")
    AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium").save_pretrained("./gpt2-medium")
    AutoTokenizer.from_pretrained("openai-community/gpt2-medium", use_fast=True).save_pretrained("./gpt2-medium")
    
    print("finished")

if __name__ == "__main__":
    download()