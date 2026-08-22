#Short code to test predictions with a T5 model

import torch
from models.t5 import PRT5, PRT5Numeric

from transformers import T5Tokenizer, T5ForConditionalGeneration

# checkpoint = "/scratch/michaelz/punc-rest-improves/outputs/checkpoints/stsb2mlm10.20251013-151830.epoch=2-val_loss=0.4909.ckpt"
# base = "t5-base"

# tokenizer = T5Tokenizer.from_pretrained(base, legacy=True)
# model = PRT5Numeric.load_from_checkpoint(checkpoint, map_location="cpu")

# model.eval()

# input_text = "(Person one said this) (Person two said that)"
# inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# with torch.no_grad():
#     outputs = model(
#       input_ids=inputs.input_ids,
#       attention_mask=inputs.attention_mask
#     )
#     prediction = outputs["preds"].item()

# print("Input Text:", input_text)
# print("Output Text:", prediction)


checkpoint = "/scratch/michaelz/punc-rest-improves/outputs/checkpoints/CoLAepoch40.20251013-131329.epoch=1-val_loss=0.2747.ckpt"
base = "t5-base"

tokenizer = T5Tokenizer.from_pretrained(base, legacy=True)
model = T5ForConditionalGeneration.from_pretrained(base)
state_dict = torch.load(checkpoint, map_location="cpu")["state_dict"]
model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})


model.eval()

input_text = "Is this sentence acceptable"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = model.generate(inputs.input_ids, max_length=512, num_beams=4, early_stopping=True)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Input Text:", input_text)
print("Output Text:", output_text)