from utils import logger, clean_split
logger = logger()


def score(texts, outputs, targets, strict: bool = False, printer = print):
    # initialize metrics
    total = 0
    correct = 0

    for sent, o, t in zip(texts, outputs, targets):
        a = [k for k in clean_split(o.lower())] # outputs
        g = clean_split(t.lower()) # gold labels
        total += 1
        if len(a) != 1 or len(g) != 1:
            print("The line has incorrect number predictions: ", sent, t)
        if a[0] == g[0]:
            correct += 1

    printer(f"Out of {total} samples, the model correctly labeled {correct} samples. \n")
    accuracy = correct/total
    printer(f"Accuracy: {accuracy}\n")

    for i in range(3):
        printer("Input text:    %s\n" % texts[i])
        printer("Actual label:    %s\n" % targets[i])
        printer("Predicted label: %s\n" % outputs[i])
        printer("=====================================================================\n")

    return accuracy