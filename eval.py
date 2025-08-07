import importlib
import json
import os

from argparse import ArgumentParser
from data.modules import TrainData
from torch.utils.data import DataLoader
from train import PRT5
from tqdm import tqdm
from typing import List, Dict, Union
from utils import logger

logger = logger()

def clean_split(s: str) -> List[str]:
    return [k.strip(' ') for k in s.strip(' ').strip(')').strip('(').strip(' ').split(') (')]

def text2triple(outputs, targets):
    output_list = []
    target_list = []
    for output, target in zip(outputs, targets):

        sentence_outputs = []
        sentence_targets = []

        if output == "":
            pass
        else:
            _os = ['(' + k.strip(')').strip('(') + ')' for k in output.split(') (')]
            for o in _os:
                output_split = o.split(';')
                if len(output_split) < 3:
                    output_split.extend(["", "", ""])
                head, pred, tail = output_split[:3]
                sentence_outputs.append({
                    "head": head.replace("(", "").strip(' '),
                    "predicate": pred.strip(' '),
                    "tail": tail.replace(")", "").strip(' ')
                })

        if target == "":
            pass
        else:
            ts = ['(' + k.strip(' ').strip(')').strip('(') + ')' for k in target.split(') (')]
            for t in ts:
                target_split = t.split(';')
                if len(target_split) < 3:
                    target_split.extend(["", "", ""])
                head, pred, tail = target_split[:3]
                sentence_targets.append({
                    "head": head.replace("(", "").strip(' '),
                    "predicate": pred.strip(' '),
                    "tail": tail.replace(")", "").strip(' ')
                })

        output_list.append(sentence_outputs)
        target_list.append(sentence_targets)

    return output_list, target_list


def sequence_tagging_score(texts, outputs, targets, printer=print):
    """
    Calculates sequence tagging score
    :param outputs: Inferred outputs
    :param targets: Gold labels
    :return: Weighted sequence tagging score
    """
    gold = 0
    correct = 0
    attempt = 0
    for output, target in zip(outputs, targets):
        os = output.split(' ')
        ts = target.split(' ')

        if len(ts) >= len(os):
            ts = ts[:len(os)]
        else:
            ts.extend(['O'] * (len(os) - len(ts)))

        assert len(os) == len(ts)

        for o, t in zip(os, ts):
            if o != 'O':
                gold += 1
                if o == t:
                    correct += 1
            if t != 'O':
                attempt += 1

    precision = correct / gold
    recall = correct / attempt
    printer(f"Correct: {correct}, gold: {gold}: precision {'%.3f' % precision}\n")
    printer(f"Correct: {correct}, attempts: {attempt}: recall {'%.3f' % recall}\n ")

    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    printer(f"F1 score: {'%.3f' % f1} \n")

    for i in range(3):
        printer("Input sentence:     %s \n" % texts[i])
        printer("Actual sentence:    %s \n" % targets[i])
        printer("Predicted sentence: %s \n" % outputs[i])
        printer("=====================================================================\n")

    return f1


def object_generation_score(texts, outputs, targets, printer=print):
    # initialize metrics
    attempts = 0
    total_gold = 0
    correct = 0

    for sent, o, t in zip(texts, outputs, targets):
        a = [k for k in clean_split(o.lower()) if k != "()"]
        g = clean_split(t.lower())
        attempts += len(a)
        total_gold += len(g)

        # use exact match
        correct += len([k for k in a if k in g])

    if 0 in [total_gold, attempts, correct]:
        printer("No accurate extractions have been made. Continuing training...\n")
        return 0

    printer(
        f"Out of {total_gold} NEs, the model extracted {correct} triples or entities correctly in {attempts} attempts.\n")
    p = correct / attempts
    r = correct / total_gold
    f1 = 2 * p * r / (p + r)
    printer(f"Precision: {p}, recall: {r}, F1 score: {f1}\n")

    for i in range(3):
        printer("Input text:    %s\n" % texts[i])
        printer("Actual NEs:    %s\n" % targets[i])
        printer("Predicted NEs: %s\n" % outputs[i])
        printer("=====================================================================\n")

    return f1


def multitask_score(texts, outputs, targets, printer=print):
    # assume NER-OIE multitask model
    ner_outputs = []
    ner_targets = []
    oie_outputs = []
    oie_targets = []

    for output, target in zip(outputs, targets):
        ner_os = []
        oie_os = []
        ner_ts = []
        oie_ts = []
        output = output.replace('() ', '')
        target = target.replace('() ', '')

        for o in clean_split(output):
            if any([k in o for k in ['LOC: ', 'PER: ', 'ORG: ', 'PRO: ']]):
                ner_os.append(f"({o})")
            else:
                oie_os.append(f"({o})")
        for t in clean_split(target):
            if any([k in t for k in ['LOC: ', 'PER: ', 'ORG: ', 'PRO: ']]):
                ner_ts.append(f"({t})")
            else:
                oie_ts.append(f"({t})")
        ner_outputs.append(' '.join(ner_os))
        ner_targets.append(' '.join(ner_ts))
        oie_outputs.append(' '.join(oie_os))
        oie_targets.append(' '.join(oie_ts))

    object_generation_score(texts, ner_outputs, ner_targets)
    object_generation_score(texts, oie_outputs, oie_targets)


def text2carb(ckpt: Union[str, None], model, input_dataset, tokenizer, max_len, batch_size,
              skip_special_tokens=True, num_beams=4,
              printer="outfile", seed=0):
    if ckpt:
        model = model.load_from_checkpoint(ckpt)
    else:
        pass

    if printer == "outfile":
        ckpt_name = sorted(ckpt.split('/'), key=lambda k: len(k), reverse=True)[0]
        printer = open(f"/data/carb/{ckpt_name}_{seed}.oie", "w").write
    elif type(printer) == str:
        printer = open(printer, 'w').write
    else:
        printer = print

    dataloader = DataLoader(input_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    model.model.eval()
    model = model.to("cuda")
    outputs = []

    for batch in tqdm(dataloader):
        if 'input_ids' in batch:
            outs = model.model.generate(input_ids=batch['input_ids'].to("cuda"),
                                        attention_mask=batch['attention_mask'].to("cuda"),
                                        max_length=max_len, num_beams=num_beams
                                        )
        else:
            outs = model.model.generate(input_ids=batch['source_ids'].to("cuda"),
                                        attention_mask=batch['source_mask'].to("cuda"),
                                        max_length=max_len, num_beams=num_beams
                                        )
        dec = [
            tokenizer.decode(ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False).strip()
            for ids in
            outs]
        outputs.extend(dec)

    structured_outputs, _ = text2triple(outputs, [""] * len(outputs))

    for t, os in zip(input_dataset.data, structured_outputs):
        printer(t + "\n")
        for o in os:
            printer(f"1.00: ({o['head']}; {o['predicate']}; {o['tail']})\n")
        printer('\n')


def pr_score(texts, outputs, targets, printer=print):
    printer(f"{len(texts)} text chunks \n")

    # calculate metric
    hit = 0
    false_alarm = 0
    miss = 0
    foul = 0
    correct_rejection = 0

    problem = 0
    attempt = 0

    len_mismatch = 0
    for source, output, target in zip(texts, outputs, targets):
        # s_tokens = source[32:].split(' ')  # prompt is 32 chars long
        s_tokens = source.split(' ')
        o_tokens = output.split(' ')
        t_tokens = target.split(' ')

        if len(s_tokens) != len(o_tokens) or len(s_tokens) != len(t_tokens):
            len_mismatch += 1
            # printer(
            #     f"Found length mismatch between source {len(s_tokens)}, output {len(o_tokens)}, target {len(t_tokens)}\n")
            # printer("Skipping...")
            min_len = min([len(s_tokens), len(o_tokens), len(t_tokens)])
            s_tokens = s_tokens[:min_len]
            o_tokens = o_tokens[:min_len]
            t_tokens = t_tokens[:min_len]
            continue

        for s, o, t in zip(s_tokens, o_tokens, t_tokens):
            # is problem
            if s != t:
                problem += 1
                if s != o:
                    attempt += 1
                    if o == t:
                        hit += 1
                    elif o != t:
                        foul += 1
                elif s == o:
                    miss += 1

            # not problem
            elif s == t:
                if s != o:
                    attempt += 1
                    false_alarm += 1
                elif s == o:
                    correct_rejection += 1

        p = hit / attempt
        r = hit / problem

    if 0 in [attempt, hit, problem]:
        print("No good extractions have been made. Continuing training...")
    else:

        printer(f"From a corpus of {problem} problems, {attempt} attempts were made. \n"
                f"Among them, we have {hit} hits, {correct_rejection} correct rejections, and {foul} fouls, "
                f"{miss} misses, {false_alarm} false alarms. \n"
                f"Precision: {p}, recall: {r}, f1: {2 * p * r / (p + r)} \n"
                f"We also report {len_mismatch} length mismatches."
                )

        # show sample
        for i in range(3):
            printer("Input sentence:     %s" % texts[i])
            printer("Actual sentence:    %s" % targets[i])
            printer("Predicted sentence: %s" % outputs[i])
            printer("=====================================================================\n")

    return 2 * p * r / (p + r)


def sbd_score(texts, outputs, targets, printer=print):
    # use macro accuracy for each tag in ["B", "I", "E"]

    gold_b, gold_i, gold_e = 0, 0, 0
    correct_b, correct_i, correct_e = 0, 0, 0
    attempt_b, attempt_i, attempt_e = 0, 0, 0

    for o, t in zip(outputs, targets):
        o = o.replace(" <EOS> ", "<EOS>")
        t = t.replace(" <EOS> ", "<EOS>")
        t_tokens = []
        for t_token in t.split(' '):
            if '<EOS>' in t_token:

                # single-word sentence take the E tag
                t_subtokens = t_token.split('<EOS>')

                gold_e += (len(t_subtokens) - 1)
                t_tokens.extend([(k, "E") for k in t_subtokens[:-1]])

                t_tokens.append((t_subtokens[-1], "B"))
                gold_b += 1


            else:
                t_tokens.append((t_token, "I"))
                gold_i += 1
        for o_token in o.split(' '):
            if '<EOS>' in o_token:
                o_subtokens = o_token.split('<EOS>')

                # single-word sentences take the E tag
                for o_subtoken in o_subtokens[:-1]:
                    attempt_e += 1
                    if (o_subtoken, "E") in t_tokens:
                        correct_e += 1

                attempt_b += 1
                if (o_subtokens[-1], "B") in t_tokens:
                    correct_b += 1

            else:
                if (o_token, "I") in t_tokens:
                    correct_i += 1
                attempt_i += 1

    # show sample
    for i in range(3):
        printer("Input sentence:     %s\n" % texts[i])
        printer("Actual sentence:    %s\n" % targets[i])
        printer("Predicted sentence: %s\n" % outputs[i])
        printer("=====================================================================\n")

    if 0 in [attempt_i, attempt_b, attempt_e]:
        print([attempt_i, attempt_b, attempt_e])
        return 0
    else:
        b_p, b_r = correct_b / attempt_b, correct_b / gold_b
        i_p, i_r = correct_i / attempt_i, correct_i / gold_i
        e_p, e_r = correct_e / attempt_e, correct_e / gold_e

        b_f1 = b_p * b_r * 2 / (b_p + b_r)
        i_f1 = i_p * i_r * 2 / (i_p + i_r)
        e_f1 = e_p * e_r * 2 / (e_p + e_r)

        macro_p = (b_p + i_p + e_p) / 3
        macro_r = (b_r + i_r + e_r) / 3
        macro_f1 = (b_f1 + i_f1 + e_f1) / 3

        printer(f"|    |  P  |  R  | F1  |\n"
                f"------------------------\n"
                f"|Begi|{'%.3f' % b_p}|{'%.3f' % b_r}|{'%.3f' % b_f1}|\n"
                f"|  In|{'%.3f' % i_p}|{'%.3f' % i_r}|{'%.3f' % i_f1}|\n"
                f"| End|{'%.3f' % e_p}|{'%.3f' % e_r}|{'%.3f' % e_f1}|\n"
                f"|Macr|{'%.3f' % macro_p}|{'%.3f' % macro_r}|{'%.3f' % macro_f1}|\n")
        return macro_f1

def run(
    task: str,
    model_name: str,
    ckpt_path: str,
    data_path: str,
    max_seq_length: int = 512,
    eval_batch_size: int = 32,
    num_workers: int = 4,
    strict: bool = True,
):
    print(f"=============== Model {model_name} {task} Evaluation ===============")
    path = 'outputs/generated/%s.jsonl' % model_name.split(' ', 1)[0]
    
    default_data_paths = {
        'pr': 'outputs/datasets/wiki-20231101.en-pr.jsonl',
        'mlm': 'outputs/datasets/wiki-20231101.en-mlm.jsonl',
        'srl': 'outputs/datasets/conll-2012-srl.jsonl',
        'pos': 'outputs/datasets/conll-2003-pos.jsonl',
        'oie': 'outputs/datasets/oie-2016-oie.jsonl',
    }
    
    texts, outputs, targets = [], [], []
    if os.path.isfile(path):
        logger.info('Restoring outputs from %s.' % path)
        with open(path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj['text'])
                outputs.append(obj['output'])
                targets.append(obj['target'])
    else:
        logger.info(f'Loading model {model_name} from checkpoint {ckpt_path}')
        model = PRT5.load_from_checkpoint(ckpt_path)
        
        logger.info(f'Loading dataset from path {data_path}')
        data_path = data_path or default_data_paths[task]
        ds = TrainData(data_path)
        
        logger.info('Initializing dataloader. ')
        dl = ds.loader(
            split = 'test',
            tokenizer = model.tokenizer,
            max_seq_length = max_seq_length,
            eval_batch_size = eval_batch_size,
            num_workers = num_workers,
        )
        
        logger.info('Generating outputs.')
        texts, outputs, targets = model.generate(dl)
        
        logger.info('Backing up outputs to %s.' % path)
        with open(path, 'w') as f:
            for i in range(len(texts)):
                text = texts[i]
                output = outputs[i] if i < len(outputs) else None
                target = targets[i] if i < len(targets) else None
                json.dump({'text': text, 'output': output, 'target': target}, f, ensure_ascii = False)
                f.write('\n')
    
    logger.info(f'Evaluating {task} score.')
    if task in ['srl', 'pos', 'oie']:
        score = importlib.import_module('tasks.' + task).score
    else:
        raise NotImplementedError(task)
    p, r, f1 = score(texts, outputs, targets, strict = strict)
    print(
        f"""
        =============== Evaluation Result ===============
        Precision: {p},
        Recall: {r},
        F1: {f1},
        """
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'task',
        type = str,
        help = 'The evaluation to perform.',
    )
    parser.add_argument(
        '-c', '--ckpt',
        type = str, required = True,
        help = 'Path to the checkpoint to be evaluated. '
    )
    parser.add_argument(
        '-d', '--dataset-jsonl',
        type = str,
        help = """
            A jsonl file containing evaluating data.
            If left unprovided, a corresponding default jsonl will be used.
            """,
    )
    parser.add_argument(
        '-n', '--model-name',
        type = str, required = True,
        help = 'Name the model that will be evaluated, to be used in result printing. '
    )
    parser.add_argument(
        '--strict',
        action = 'store_true',
        help = 'Use a stricter metric for evaluation. See implementation detail for each task.'
    )
    args = parser.parse_args()
    
    run(
        task = args.task,
        model_name = args.model_name,
        ckpt_path = args.ckpt,
        data_path = args.dataset_jsonl,
        strict = args.strict,
    )
