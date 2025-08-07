# To fine tune to baseline run:
python main.py --model_name_or_path t5-base --output_dir outputs --NER_eval_flag --dataset_name processed_conll03.jsonl

# To fine tune on the PR extended training model run:
python main.py --model_name_or_path trained_PR_model --output_dir outputs --NER_eval_flag --dataset_name processed_conll03.jsonl