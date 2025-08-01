import re

from utils import multiset_intersection, prf1, logger

logger = logger()


def seq_to_struct_lax(s: str) -> set[tuple[str, str]]:
    """
    Given a sequence, return a set of three tuples of the role-labelled words,
    with the verb of their respective frame and the role label.

    For example: the string 'eat (A: ham burger, B: chicken), drink (C: coke sprite beer sprite)' will return
    {
        (ham, eat, A), (burger, eat, A), (chicken, eat, B),
        (coke, drink, C), (sprite, drink, C), (beer, drink, C),
    }
    """
    out: set[tuple[str, str]] = set()
    for verb_frame in re.finditer(r'(\S+) \((.*?\))', s.strip(' ,')):
        for label_friends in re.finditer(r'([A-Z\-\d]+): (.*?\S)[,)]', verb_frame.group(2)):
            label, friends = label_friends.group(1), label_friends.group(2)
            for friend in friends.split():
                out.add((friend, label))
    return out


def seq_to_struct(s: str) -> tuple[dict[str, dict[str, dict[str, int]]], int]:
    """
    Given a sequence, divide it into verb frames,
    then for each verb, list the semantic roles and words in the same sentence with that role in a multiset,
    along with the total number of semantic role labels,

    For example: the string 'eat (A: ham burger, B: chicken), drink (C: coke sprite beer sprite)' will return
    {
        'eat_1': {
            'A': {'ham': 1, 'burger': 1},
            'B': {'chicken': 1},
        },
        'drink_1': {
            'C': {'coke': 1, 'sprite': 2, 'beer': 1},
        },
    },
    7
    """
    out: dict[str, dict[str, dict[str, int]]] = {}
    verb_counter: dict[str, int] = {}
    total_labels: int = 0
    for verb_frame in re.finditer(r'(\S+) \((.*?\))', s.strip(' ,')):
        verb, bracket_content = verb_frame.group(1), verb_frame.group(2)
        verb_counter.setdefault(verb, 0)
        verb_counter[verb] += 1
        
        verb = f'{verb}_{verb_counter[verb]}'
        out.setdefault(verb, {})
        verb_dict = out[verb]
        for label_friends in re.finditer(r'([A-Z\-\d]+): (.*?\S)[,)]', bracket_content):
            label, friends = label_friends.group(1), label_friends.group(2)
            verb_dict.setdefault(label, {})
            for friend in friends.split():
                friend = friend.strip(' ')
                verb_dict[label].setdefault(friend, 0)
                verb_dict[label][friend] += 1
                total_labels += 1
    
    return out, total_labels


def score(texts: list[str], outputs: list[str], targets: list[str], distinguish_verb_frames: bool) \
    -> tuple[float, float, float]:
    """Calculate precision, recall, and F1 score for SRL task on CoNLL 2012"""
    num_correct, num_attempted, num_gold = 0, 0, 0
    logger.debug('Evaluating. Imperfect attempts will be logged. \n')
    for text, output, target in zip(texts, outputs, targets):
        if distinguish_verb_frames:
            output_dict, output_label_count = seq_to_struct(output)
            num_attempted += output_label_count
            
            target_dict, target_label_count = seq_to_struct(target)
            num_gold += target_label_count
            
            if output_dict == target_dict:
                num_correct += target_label_count
                continue
            
            acc, mislabeled = 0, 0
            for verb in output_dict.keys() & target_dict.keys():
                output_verb_frame = output_dict[verb]
                target_verb_frame = target_dict[verb]
                for label in output_verb_frame.keys() & target_verb_frame.keys():
                    acc += multiset_intersection(
                        output_verb_frame[label],
                        target_verb_frame[label],
                    )
                for label in output_verb_frame.keys() - target_verb_frame.keys():
                    output_frame_elements = output_verb_frame[label]
                    for _, target_frame_elements in target_verb_frame.items():
                        if output_frame_elements == target_frame_elements:
                            # acc += len(output_frame_elements) * 1 / 2 # Partial Scores
                            mislabeled += len(output_frame_elements)
            print(
                f"""
                =============== Incorrect Output ===============
                Text:   {text}
                Output: {output}
                Target: {target}
                =============== Scoring ===============
                {acc} with {output_label_count} attempted and {target_label_count} gold; {mislabeled} mislabeled.

                """
            )
            num_correct += acc
            acc, mislabeled = 0, 0
        else:
            output_set = seq_to_struct_lax(output)
            num_attempted += len(output_set)
            
            target_set = seq_to_struct_lax(target)
            num_gold += len(target_set)
            
            num_correct += len(output_set & target_set)
    
    return prf1(num_correct, num_attempted, num_gold)