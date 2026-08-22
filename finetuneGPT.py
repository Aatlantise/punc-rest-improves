import os
import re
import torch
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments
)

MODEL_NAME = "gpt2"
MAX_LEN = 128
PUNCTUATION_TO_REMOVE = {',', '.', '!', '?', '"', "'", ":", ";"}
batch_size = 16
epochs = 3

def normalize_text(text):
    """Lowercase and remove specific punctuation and capitalization."""
    text = text.lower()
    result = ''.join(ch for ch in text if ch not in PUNCTUATION_TO_REMOVE)
    result = re.sub('\s+', ' ', result)
    return result

print("load yelp")
dataset = load_dataset("yelp_review_full")
train_data = dataset["train"]
test_data = dataset["test"]

print("load model")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def preprocess(example):
    source = normalize_text(example["text"])
    target = example["text"]
    
    prefix = "restore punctuation: " + source
    
    # full_text = "restore punctuation: " + input + tokenizer.eos_token + target + tokenizer.eos_token
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    
    input_ids = prefix_ids+target_ids
    labels = [-100]*len(prefix_ids) + target_ids
    
    input_ids = input_ids[:MAX_LEN]
    labels = labels[:MAX_LEN]
    
    attention_mask = [1]*len(input_ids)
    
    padding = MAX_LEN - len(input_ids)
    if padding > 0:
      input_ids = input_ids + [tokenizer.pad_token_id]*padding
      labels = labels + [-100] * padding
      attention_mask = attention_mask + [0]*padding
    
    return {"input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            }
    # tokens = tokenizer(full_text,truncation = True,padding = "max_length",max_length = MAX_LEN)
    # tokens["labels"] = tokens["input_ids"].copy()
    # return tokens

tokenized = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.config.pad_token_id = tokenizer.pad_token_id

save_name = "gpt2_yelp_pr"
save_dir = os.path.join("checkpoints", save_name)
os.makedirs(save_dir, exist_ok=True)

training_args = TrainingArguments(
        output_dir=save_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        fp16=torch.cuda.is_available(),  # automatic mixed precision on GPU
    )

trainer = Trainer(
    model = model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"]
)

print("train")
trainer.train()


model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"saved to {save_dir}")