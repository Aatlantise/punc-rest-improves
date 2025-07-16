from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Union

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


def triple_soft_match_score(texts: List[str],
                            output_list: List[List[Dict]],
                            target_list: List[List[Dict]],
                            outfile=None,
                            printer=print):
    """
    Calculates various OIE metrics including Handcrafted_gold scores, exact matche F1.

    :param outfile:
    :param texts: List of test set sentences
    :param output_list: List of output dictionaries in the form of {"head": str, "predicate": str, "tail": str}
    :param target_list: List of target (gold) dictionaries in the same form as output_list
    :param printer: Printing function--can be a logger or print, or other custom function

    :return: None
    """

    def evaluate(sent, gold_head, gold_pred, gold_tail, extractions):
        '''
        Evaluates predicate candidates and selects best one for given gold triple

        Args:
            sent: Original CaRB sentence in string format
            gold_head: Gold or original CaRB head in string format
            gold_pred: Gold or original CaRB predicate in string format
            gold_tail: Gold or original CaRB tail in string format
            extractions: List of extracted triples
            [
                {
                    "head": "This",
                    "predicate": "is",
                    "tail": "an example sentence"
                },
                ...
            ]

        Returns:
            Tuple of strings in the form of (judgement, best predicate)
        '''
        candidates = {}
        for extraction in extractions:
            extracted_pred = extraction['predicate']
            if extraction is None:
                continue
            elif is_same(gold_pred, extracted_pred):
                return "good", extraction
            elif overlap_rate(gold_pred, extraction['predicate']) < 0.25 or extracted_pred == '-':
                continue
            elif not is_sequential(sent, gold_head, gold_pred, gold_tail, extracted_pred):
                # if nonseq
                if diff_is_adv(gold_pred, extracted_pred):
                    # if diff is adv then acceptable
                    if "acceptable" in candidates:
                        candidates["acceptable"].append(extraction)
                    else:
                        candidates["acceptable"] = [extraction]
                else:
                    # else incorrect
                    if "incorrect" in candidates:
                        candidates["incorrect"].append(extraction)
                    else:
                        candidates["incorrect"] = [extraction]
            elif overlap_rate(gold_head, extracted_pred) > 0.75 or \
                    overlap_rate(gold_tail, extracted_pred) > 0.75:
                # also incorrect if trespass head / tail boundary too much
                if "acceptable" in candidates:
                    candidates["acceptable"].append(extraction)
                else:
                    candidates["acceptable"] = [extraction]
            else:  # no issues then acceptable, I guess
                if "acceptable" in candidates:
                    candidates["acceptable"].append(extraction)
                else:
                    candidates["acceptable"] = [extraction]

        # loop over candidates and their judgements
        if "acceptable" in candidates:
            return "acceptable", shortest_extraction(candidates["acceptable"])
        elif "incorrect" in candidates:
            return "incorrect", shortest_extraction(candidates["incorrect"])
        else:
            return "no detection", {"head": "-", "predicate": "-", "tail": "-"}

    def diff_is_adv(gold_pred, extraction):
        '''
        Determines if the extraction is only missing a single adverb from gold predicate.

        Args:
            gold_pred
            extraction

        Returns:
            A boolean argument
        '''
        advs = ["afterward", "already", "almost", "back", "better", "best", "even", "far", "fast", "hard",
                "here", "how", "late", "long", "low", "more", "near", "never", "next", "now", "often",
                "quick", "rather", "slow", "so", "soon", "still", "then", "today", "tomorrow", "too", "very", "well",
                "where", "yesterday"]
        if overlap_rate(extraction, gold_pred) != 1:
            # extraction not entirely in gold_pred
            return False
        diff = []
        ext_words = extraction.split(' ')
        for gold_word in gold_pred.split(' '):
            if gold_word not in ext_words:
                diff.append(gold_word)
        if len(diff) != 1:
            return False
        diff = diff[0]
        if diff[-2:] == 'ly' or diff in advs:
            return True
        else:
            return False

    def is_same(gold_pred, extraction):
        '''
        Whether gold predicate and extracted predicate are equal
        '''
        if gold_pred.replace(' ', '').lower() == extraction.replace(' ', '').lower():
            return True
        else:
            return False

    def overlap_rate(this, that):
        '''
        How much of this is also in that?

        Args:
            this: denominator
            that: numerator

        Returns:
            In decimal form how much of this is also in that
        '''
        this_words = this.split(' ')
        this_len = len(this_words)
        that_words = that.split(' ')
        overlap = 0
        for this_word in this_words:
            if this_word in that_words:
                overlap += 1
        return overlap / this_len

    def shortest_extraction(extractions: List[Dict]) -> Dict:
        # use dummy dictionary
        shortest = {"predicate": ""}
        shortest_len = 0

        # loop over extraction dictionaries
        for extraction in extractions:
            extracted_pred = extraction['predicate']
            ext_len = len(extracted_pred.split(' '))
            if shortest['predicate'] == '' or ext_len < shortest_len:
                shortest = extraction
                shortest_len = ext_len
            else:
                continue
        return shortest

    def is_sequential(sentence, gold_head, gold_pred, gold_tail, extraction):
        ext_nosp = extraction.replace(' ', '').lower()
        if ext_nosp in sentence.replace(' ', '').lower():
            return True
        elif ext_nosp in (gold_head + gold_pred + gold_tail).replace(' ', '').lower():
            return True
        else:
            return False

    # compare output and targets
    head_exact_match = 0
    tail_exact_match = 0
    pred_exact_match = 0
    triple_exact_match = 0
    incorrect_inferences = []
    num_attempts = sum([len(k) for k in output_list])
    num_gold = sum([len(k) for k in target_list])
    printer(f"Extracted {num_attempts} extractions from {len(texts)} sentences. Gold set has "
            f"{num_gold} extractions.\n")
    if outfile is not None:
        f = open(outfile, 'w')
        f.write('\t'.join(["sentence", "head", "pred", "tail"]) + "\n")
    judgements = {"good": 0, "acceptable": 0, "incorrect": 0, "no detection": 0}
    for sentence, sentence_outputs, sentence_targets in zip(texts, output_list, target_list):

        # loop over target and outputs
        for t in sentence_targets:
            gold_head = t['head']
            gold_pred = t['predicate']
            gold_tail = t['tail']
            judgement, silver = evaluate(sentence, gold_head, gold_pred, gold_tail, sentence_outputs)
            judgements[judgement] += 1

            # record extractions
            if outfile is not None:
                f.write('\t'.join([sentence, gold_head, gold_pred, gold_tail, judgement,
                                   silver["head"], silver["predicate"], silver["tail"]]) + "\n")

            # count exact matches
            triplewise_match = 0
            if gold_head in [s['head'] for s in sentence_outputs]:
                head_exact_match += 1
                triplewise_match += 1
            if gold_pred in [s['predicate'] for s in sentence_outputs]:
                pred_exact_match += 1
                triplewise_match += 1
            if gold_tail in [s['tail'] for s in sentence_outputs]:
                tail_exact_match += 1
                triplewise_match += 1
            if triplewise_match == 3:
                triple_exact_match += 1

    # calculate metrics
    score = (judgements['good'] + 0.5 * judgements['acceptable'] - judgements['incorrect']) / sum(judgements.values())
    printer(f"Model finds {judgements['good']} good extractions, {judgements['acceptable']} acceptable extractions,"
            f" {judgements['incorrect']} incorrect extractions, and {judgements['no detection']} no detections "
            f"to achieve a score of {score}.'\n")
    printer(f"Number of exactly matched heads is {head_exact_match},"
            f" predicates {pred_exact_match}, tails {tail_exact_match}, and triples {triple_exact_match}\n")

    if 0 in [num_attempts, num_gold]:
        print("No good extractions have been made. Continuing training...")
        return 0
    else:
        ps = [k / num_attempts for k in [head_exact_match, pred_exact_match, tail_exact_match, triple_exact_match]]
        rs = [k / num_gold for k in [head_exact_match, pred_exact_match, tail_exact_match, triple_exact_match]]
        f1 = [2 * p * r / (p + r) for p, r in zip(ps, rs)]
        printer(f"|    |  P  |  R  | F1  |\n"
                f"------------------------\n"
                f"|Head|{'%.3f' % (ps[0])}|{'%.3f' % (rs[0])}|{'%.3f' % (f1[0])}|\n"
                f"|Pred|{'%.3f' % (ps[1])}|{'%.3f' % (rs[1])}|{'%.3f' % (f1[1])}|\n"
                f"|Tail|{'%.3f' % (ps[2])}|{'%.3f' % (rs[2])}|{'%.3f' % (f1[2])}|\n"
                f"| Tot|{'%.3f' % (ps[3])}|{'%.3f' % (rs[3])}|{'%.3f' % (f1[3])}|\n")

        # show examples
        for i in incorrect_inferences:
            printer(f"sentence: {i[0][35:]}")
            printer(f"gold extractions: {i[1]}")
            printer(f"predicted extractions: {i[2]}\n\n")

        return f1[1]  # return predicate f1


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

# Calculate the recall, precision and f1 score for NER tasks
def NER_eval(texts, outputs, targets, printer=print):
    attempts = 0
    total_gold = 0
    correct = 0

    def clean_space(s:str):
        # return [k.strip(' ') for k in s.strip(' ').strip(')').strip('(').strip(' ').split(') (')]
        return [k for k in s.split(' ') if k != '']
    
    for sent, o, t in zip(texts, outputs, targets):
        a = [k for k in clean_space(o.lower()) if k != "()"]
        # ignore empty strings "()"
        g = clean_space(t.lower())
        attempts += len(a) # total number of predictions
        total_gold += len(g) # total number of labels
        print("input: ", sent)
        print("predicted labels: ", a)
        print("gold labels: ", g)

        # use exact match
        correct += len([k for k in a if k in g])
    
    if 0 in [total_gold, attempts, correct]:
        printer("No accurate extractions have been made. Continuing training...\n")
        return 0
    
    printer(f"Out of {total_gold} NEs, the model extracted {correct} triples or entities correctly in {attempts} attempts.\n")
    p = correct / attempts
    r = correct / total_gold
    f1 = 2 * p * r / (p + r)
    printer(f"Precision: {p}, recall: {r}, F1 score: {f1}\n")
    
    printer("NER scores below:\n")
    for i in range(3):
        printer("Input text:    %s\n" % texts[i])
        printer("Actual NEs:    %s\n" % targets[i])
        printer("Predicted NEs: %s\n" % outputs[i])
        printer("=====================================================================\n")

    return f1