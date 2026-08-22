import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch

VERBOSE = True

def evluate_hans(model_path):
    print(f"\n===== Evaluating model from {model_path} on HANS =====")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    hans = load_dataset("hans", split="validation")
    heuristics = ["lexical_overlap", "subsequence", "constituent"]
    subcases = ["non-entailment", "entailment"]
    results = {h: {s: [] for s in subcases} for h in heuristics}
    
    sample_errors = {h: {s: [] for s in subcases} for h in heuristics}
    MAX_SAMPLES = 5

    for example in hans:
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        heuristic = example["heuristic"]
        gold = example["label"]
        subtype = "entailment" if gold == 0 else "non-entailment"

        inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding="max_length").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = np.argmax(logits.cpu().numpy())
            #convert hans label, 0:entailment is 0 in hans, 1,2: non-entailment is 1 in hans
            predicted_hans_label = 0 if predicted_class == 0 else 1
            
            correct = int(predicted_hans_label == gold)
            
            if VERBOSE and correct == 0 and len(sample_errors[heuristic][subtype]) < MAX_SAMPLES:
                sample_errors[heuristic][subtype].append({
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "gold": gold,
                    "predicted_hans": predicted_hans_label,
                    "predicted_mnli": predicted_class
                })
        
            results[heuristic][subtype].append(int(predicted_hans_label == gold))
        
    final_results = {}
    print("\n===== HANS Evaluation Results =====\n")
    for h in heuristics:
        final_results[h] = {}
        for s in subcases:
            arr = results[h][s]
            acc = (sum(arr) / len(arr)) * 100 if len(arr) > 0 else float('nan')
            final_results[h][s] = acc
            print(f"Heuristic: {h}, Subcase: {s}, Accuracy: {acc:.2f}%")
    if VERBOSE:
      print("\n===== SAMPLE ERRORS (up to 5 per category) =====\n")
      for h in heuristics:
          for s in subcases:
              errs = sample_errors[h][s]
              if not errs:
                  continue
              print(f"\n--- {h} / {s} ---")
              for e in errs:
                  print(f"Premise:    {e['premise']}")
                  print(f"Hypothesis: {e['hypothesis']}")
                  print(f"Gold: {e['gold']}, Pred(HANS): {e['predicted_hans']}, Pred(MNLI): {e['predicted_mnli']} \n")
    return final_results

if __name__ == "__main__":
    model_paths = ["checkpoints/bert-base-uncased_mnli_done", 
                   "checkpoints/felflare_bert-restore-punctuation_mnli_done"]

    all_results = {}

    for model_path in model_paths:
        scores = evluate_hans(model_path)
        all_results[model_path] = scores
    
    print("\n======== FINAL RESULTS ========")
    heuristics = ["lexical_overlap", "subsequence", "constituent"]
    subcases = ["non-entailment", "entailment"]

    for model_path in model_paths:
        print(f"\nResults for model: {model_path}")
        for h in heuristics:
            for s in subcases:
                acc = all_results[model_path][h][s]
                print(f"Heuristic: {h}, Subcase: {s}, Accuracy: {acc:.6f}%")
