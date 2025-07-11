from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

trained_model_path = "/home/michaelz/punc-rest-improves/outputs/checkpoints/epoch=1-val_loss=0.0936.ckpt"
directory = "trained_PR_model"
base = "t5-base"

model = T5ForConditionalGeneration.from_pretrained(base)
state_dict = torch.load(trained_model_path, map_location="cpu")["state_dict"]

model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})
model.save_pretrained(directory)

tokenizer = T5Tokenizer.from_pretrained(base, legacy=True)
tokenizer.save_pretrained(directory)


print(f"pretrained punctuation model saved to {directory}")