import re
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

MODEL_DIR = "checkpoints/gpt2_yelp_pr"
PUNCTUATION_TO_REMOVE = {',', '.', '!', '?', '"', "'", ":", ";"}

def normalize_text(text):
    """Lowercase and remove specific punctuation and capitalization."""
    text = text.lower()
    result = ''.join(ch for ch in text if ch not in PUNCTUATION_TO_REMOVE)
    result = re.sub('\s+', ' ', result)
    return result


tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
model.eval()


text = "Some random words. Here's punctuation!"
clean = normalize_text(text)

prompt = f"restore punctuation: {clean}"

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
    )

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded[len(prompt):])
