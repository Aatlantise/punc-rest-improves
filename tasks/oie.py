import re

from utils import prf1, logger, getstr_safe

logger = logger()


def oie_dict_count(a: dict[str, dict[str, set[str]]]) -> int:
    counter = 0
    for _, v1 in a.items():
        for _, v2 in v1.items():
            counter += len(v2)
    return counter


def oie_dict_intersection(a: dict[str, dict[str, set[str]]], b: dict[str, dict[str, set[str]]]) -> int:
    counter = 0
    for k1 in a.keys() & b.keys():
        for k2 in a[k1].keys() & b[k1].keys():
            counter += len(a[k1][k2] & b[k1][k2])
    return counter
    

def unserialize(example: str, strict: bool) -> dict[str, dict[str, set[str]]]:
    """
    Given an OIE formated output sequence,
    split it into clauses into a dictionary indexed by the subject and the verb,
    followed by a set of subordinate clauses.
    
    For example, the following string
    `(Alice; has; the key) (Alice; sleep) (Bob; runs; fast; towards the school)`
    will output
    {
        'Alice': {
            'has': {'the key'},
            'sleep': {},
        },
        'Bob': {
            'runs': {'fast', 'towards the school'},
        },
    }
    """
    out: dict[str, dict[str, set[str]]] = {}
    for clause in re.finditer(r'\((.+?\))', example):
        splits = clause.group(1).split(';')
        components = [comp.strip(' )') for comp in splits]
        if len(components) < 2:
            continue
        
        subject = components[0].strip()
        verb = components[1].strip()
        # Ω is a placeholder to signify that no object is provided for this clause, counts for 1 point
        remaining_parts = [a.strip() for a in components[2:]] if len(components) > 2 else ['Ω']
        
        verb_words = verb.split(' ')
        if len(verb_words) > 1 and not strict:
            # make verb just one word to avoid the second part including prepositions and avoiding a match
            verb = verb_words[0]
            remaining_parts.extend(verb_words[1:])
        
        if len(components) > 2 and not strict:
            new_splits = []
            for phrase in remaining_parts:
                new_splits.extend(phrase.split(' '))
            remaining_parts = new_splits
        
        out.setdefault(subject, {})
        out[subject].setdefault(verb, set())
        out[subject][verb] = out[subject][verb].union(set(remaining_parts))
        
    return out


def triple_soft_match_score(
    texts: list[str],
    outputs: list[str],
    targets: list[str],
    outfile = None,
    printer = print
) -> tuple[float, float, float]:
    """
    Calculates various OIE metrics including Handcrafted_gold scores, exact matches F1.

    :param outfile:
    :param texts: List of test set sentences
    :param outputs: List of output sequences
    :param targets: List of target (gold) sequences
    :param printer: Printing function--can be a logger or print, or other custom function
    """
    output_list: list[list[dict[str, str]]] = []
    for output in outputs:
        temp = []
        for paren in re.finditer(r'\((.+?)\)', output):
            components = paren.group(1).split(';')
            temp.append({
                'head': getstr_safe(components, 0),
                'predicate': getstr_safe(components, 1),
                'tail': getstr_safe(components, 2),
            })
        output_list.append(temp)
    
    target_list: list[list[dict[str, str]]] = []
    for target in targets:
        temp = []
        for paren in re.finditer(r'\((.+?)\)', target):
            components = paren.group(1).split(';')
            temp.append({
                'head': getstr_safe(components, 0),
                'predicate': getstr_safe(components, 1),
                'tail': getstr_safe(components, 2),
            })
        target_list.append(temp)

    def evaluate(sent, gold_head, gold_pred, gold_tail, extractions):
        """
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
        """
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
        """
        Determines if the extraction is only missing a single adverb from gold predicate.

        Args:
            gold_pred
            extraction

        Returns:
            A boolean argument
        """
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
        """
        Whether gold predicate and extracted predicate are equal
        """
        if gold_pred.replace(' ', '').lower() == extraction.replace(' ', '').lower():
            return True
        else:
            return False

    def overlap_rate(this, that):
        """
        How much of this is also in that?

        Args:
            this: denominator
            that: numerator

        Returns:
            In decimal form how much of this is also in that
        """
        this_words = this.split(' ')
        this_len = len(this_words)
        that_words = that.split(' ')
        overlap = 0
        for this_word in this_words:
            if this_word in that_words:
                overlap += 1
        return overlap / this_len

    def shortest_extraction(extractions: list[dict]) -> dict:
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

    return triple_exact_match, num_attempts, num_gold


def score(texts: list[str], outputs: list[str], targets: list[str], strict = False) -> tuple[float, float, float]:
    """Score OIE by matching"""
    if strict:
        num_correct, num_attempted, num_gold = 0, 0, 0
        for text, output, target in zip(texts, outputs, targets):
            output_clauses, target_clauses = unserialize(output, strict), unserialize(target, strict)
            
            attempted = oie_dict_count(output_clauses)
            gold = oie_dict_count(target_clauses)
            correct = oie_dict_intersection(output_clauses, target_clauses)
            
            num_attempted += attempted
            num_gold += gold
            num_correct += correct
            
            # logger.debug('\n')
    else:
        num_correct, num_attempted, num_gold = triple_soft_match_score(texts, outputs, targets)
        
    return prf1(num_correct, num_attempted, num_gold)


if __name__ == '__main__':
    s = '(The second; titled;  Consider Her Ways '') (Her Ways; consider) (The second; starred; as the lead named Jane Waterleigh; Barrie) ( Consider Her Ways ''; starred; as the lead named Jane Waterleigh; Barrie) (the lead; named; Jane Waterleigh)'
    print(unserialize(s, strict = False))