from utils import logger, clean_split
from scipy.stats import pearsonr
logger = logger()


def score(texts, outputs, targets, strict: bool = False, printer = print):
    # initialize metrics
    predictions = 0
    golds = 0
    total = 0
    failed = 0

    for sent, o, t in zip(texts, outputs, targets):
        total += 1
        printer("gold is: ", t)
        printer("pred is: ", o)
        try:
            pred = float(o)
            gold = float(t)
            predictions.append(pred)
            golds.append(gold)
        except ValueError:
            print("The line has malformed predictions: ", sent, "out: ", o, "target: ", t)
            failed += 1
            total -= 1


    printer(f"Out of {total} samples, the model failed {failed} samples. \n")
    pearson = pearsonr(predictions, golds)[0]
    printer(f"Pearson correlation score: {pearson}\n")

    for i in range(3):
        printer("Input text:    %s\n" % texts[i])
        printer("Actual label:    %s\n" % targets[i])
        printer("Predicted label: %s\n" % outputs[i])
        printer("=====================================================================\n")

    return pearson