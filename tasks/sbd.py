from utils import logger

logger = logger(__name__)


def score(texts, outputs, targets, printer = print):
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
        return 0, 0, 0
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
        return macro_p, macro_r, macro_f1