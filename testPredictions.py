#Short code to test predictions with a T5 model

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

checkpoint = "/home/michaelz/punc-rest-improves/outputs/checkpoints/epoch=1-val_loss=0.0936.ckpt"
base = "t5-base"

tokenizer = T5Tokenizer.from_pretrained(base, legacy=True)
model = T5ForConditionalGeneration.from_pretrained(base)
state_dict = torch.load(checkpoint, map_location="cpu")["state_dict"]
model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})

model.eval()

input_text = "Learn some full stops another sentence it was her cats collar its food bowl is red"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = model.generate(inputs.input_ids, max_length=512, num_beams=4, early_stopping=True)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Input Text:", input_text)
print("Output Text:", output_text)



input_text = "the nellie a cruising yawl swung to her anchor without a flutter of the sails and was at rest the flood had made the wind was nearly calm and being bound down the river the only thing for it was to come to and wait for the turn of the tide "
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = model.generate(inputs.input_ids, max_length=512, num_beams=4, early_stopping=True)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Input Text:", input_text)
print("Output Text:", output_text)
