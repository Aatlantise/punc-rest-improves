from utils import logger, prf1

logger = logger()


def score(texts, outputs, targets, strict = False, printer = print):
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

    precision, recall, f1 = prf1(correct, attempt, gold)
    printer(f"Correct: {correct}, gold: {gold}: precision {'%.3f' % precision}\n")
    printer(f"Correct: {correct}, attempts: {attempt}: recall {'%.3f' % recall}\n ")
    printer(f"F1 score: {'%.3f' % f1} \n")

    for i in range(3):
        printer("Input sentence:     %s \n" % texts[i])
        printer("Actual sentence:    %s \n" % targets[i])
        printer("Predicted sentence: %s \n" % outputs[i])
        printer("=====================================================================\n")

    return precision, recall, f1