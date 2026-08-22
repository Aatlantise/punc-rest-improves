#Short code to test predictions with a T5 model

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

checkpoint = "/scratch/michaelz/punc-rest-improves/outputs/checkpoints/T5PR1e4LR-20250825-012033-epoch40.ckpt"
base = "t5-base"

tokenizer = T5Tokenizer.from_pretrained(base, legacy=True)
model = T5ForConditionalGeneration.from_pretrained(base)
state_dict = torch.load(checkpoint, map_location="cpu")

model.eval()

input_text = "Learn some full stops another sentence it was her cats collar its food bowl is red"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = model.generate(inputs.input_ids, max_length=512, num_beams=4, early_stopping=True)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Input Text:", input_text)
print("Output Text:", output_text)



input_text = "(,)"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = model.generate(inputs.input_ids, max_length=512, num_beams=4, early_stopping=True)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Input Text:", input_text)
print("Output Text:", output_text)


input_text = "()"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = model.generate(inputs.input_ids, max_length=512, num_beams=4, early_stopping=True)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Input Text:", input_text)
print("Output Text:", output_text)
