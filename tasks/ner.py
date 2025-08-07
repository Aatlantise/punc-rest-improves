from utils import logger, clean_split, prf1

logger = logger()


def process(tokens: list[str], tags: list[str]) -> tuple[str, str]:
    source = ' '.join(tokens)
    target = []
    current = ""
    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            # if it starts with B- then this is the start of new phrase
            if current:
                target.append(current)
            current = f"({token}:{tag[2:]})"
        elif tag.startswith("I-"):
            # if it starts with I- then word is inside a phrase
            # 2 for the I- and 1 for the ":" and 1 for the ")"
            current = current[:-len(tag[2:]) - 2] + f" {token}:{tag[2:]})"
        else:
            if current:
                target.append(current)
            current = ""
    if current:
        target.append(current)
    return source, ' '.join(target) if target else "O"  # O for no prediction


def score(texts, outputs, targets, strict: bool = False, printer = print):
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
    p, r, f1 = prf1(correct, attempts, total_gold)
    printer(f"Precision: {p}, recall: {r}, F1 score: {f1}\n")

    for i in range(3):
        printer("Input text:    %s\n" % texts[i])
        printer("Actual NEs:    %s\n" % targets[i])
        printer("Predicted NEs: %s\n" % outputs[i])
        printer("=====================================================================\n")

    return p, r, f1