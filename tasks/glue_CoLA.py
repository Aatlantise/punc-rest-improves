from utils import logger, clean_split
logger = logger()


# def score(texts, outputs, targets, strict: bool = False, printer = print):
#     # initialize metrics
#     total = 0
#     correct = 0

#     for sent, o, t in zip(texts, outputs, targets):
#         a = [k for k in clean_split(o.lower())] # outputs
#         g = clean_split(t.lower()) # gold labels
#         total += 1
#         if len(a) != 1:
#             print("The line has incorrect number predictions: ", sent)
#             continue # don't want to mark as correct
#         if len(g) != 1:
#             print("The gold label is not formatted correctly", t)
#             total -=1 # ignore this sample
#             continue # don't want to mark as correct

#         # use exact match
#         if a[0] == g[0]:
#             correct += 1

#     printer(
#         f"Out of {total} samples, the model correctly labeled {correct} samples. \n")
#     accuracy = correct/total
#     printer(f"Accuracy: {accuracy}\n")

#     for i in range(3):
#         printer("Input text:    %s\n" % texts[i])
#         printer("Actual label:    %s\n" % targets[i])
#         printer("Predicted label: %s\n" % outputs[i])
#         printer("=====================================================================\n")

#     return accuracy



def score(texts, outputs, targets, strict: bool = False, printer = print):
    # initialize metrics
    correct = 0
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0
    malformed = 0
    total = 0

    for sent, o, t in zip(texts, outputs, targets):
        total += 1
        a = [k for k in clean_split(o.lower())] # outputs
        printer("gold is: ", t)
        printer("pred is: ", o)
        g = clean_split(t.lower()) # gold labels
        if len(a) != 1 or len(g) != 1:
            print("The line has incorrect number predictions: ", sent, t)
            malformed += 1
            continue # don't want to mark as correct

        # use exact match
        if a[0] == g[0]:
            correct += 1
            truePositive += 1 if a[0] == "acceptable" else 0
            trueNegative += 1 if a[0] == "unacceptable" else 0
        else:
            falsePositive += 1 if a[0] == "acceptable" else 0
            falseNegative += 1 if a[0] == "unacceptable" else 0

    printer(f"Out of {total} samples, the model correctly labeled {correct} samples. \n")
    denom = (truePositive + falsePositive) * (truePositive + falseNegative) * (trueNegative + falsePositive) * (trueNegative + falseNegative)
    MCC = 0.0
    if denom != 0:
        MCC = (truePositive * trueNegative - falsePositive * falseNegative) / (denom ** 0.5)

    for i in range(3):
        printer("Input text:    %s\n" % texts[i])
        printer("Actual label:    %s\n" % targets[i])
        printer("Predicted label: %s\n" % outputs[i])
        printer("=====================================================================\n")

    return MCC