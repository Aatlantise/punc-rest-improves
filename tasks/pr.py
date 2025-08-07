from utils import logger, prf1

logger = logger()


def score(texts, outputs, targets, printer = print, strict = False):
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

    p, r, f1 = prf1(problem, attempt, hit)

    if 0 in [attempt, hit, problem]:
        print("No good extractions have been made. Continuing training...")
    else:
        printer(f"From a corpus of {problem} problems, {attempt} attempts were made. \n"
                f"Among them, we have {hit} hits, {correct_rejection} correct rejections, and {foul} fouls, "
                f"{miss} misses, {false_alarm} false alarms. \n"
                f"Precision: {p}, recall: {r}, f1: {f1} \n"
                f"We also report {len_mismatch} length mismatches."
                )

        # show sample
        for i in range(3):
            printer("Input sentence:     %s" % texts[i])
            printer("Actual sentence:    %s" % targets[i])
            printer("Predicted sentence: %s" % outputs[i])
            printer("=====================================================================\n")

    return p, r, f1