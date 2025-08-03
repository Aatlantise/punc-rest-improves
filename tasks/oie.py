import re

from utils import prf1, logger

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
    

def unserialize(example: str) -> dict[str, dict[str, set[str]]]:
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
    for clause in re.finditer(r'\((.+?)\)', example):
        components = clause.group(1).split(';')
        subject = components[0].strip()
        if len(components) < 2:
            logger.warning(f'Bad clause split! Clause looks like')
            logger.warning(clause)
            verb = "ß"
        else:
            verb = components[1].strip()
        
        # Ω is a placeholder to signify that no object is provided for this clause, counts for 1 point
        subordinates = [a.strip() for a in components[2:]] if len(components) > 2 else ['Ω']
        
        out.setdefault(subject, {})
        out[subject].setdefault(verb, set())
        out[subject][verb] = out[subject][verb].union(set(subordinates))
        
    return out


def score(texts: list[str], outputs: list[str], targets: list[str], strict = True) -> tuple[float, float, float]:
    """Score OIE by matching"""
    num_correct, num_attempted, num_gold = 0, 0, 0
    for text, output, target in zip(texts, outputs, targets):
        output_clauses, target_clauses = unserialize(output), unserialize(target)
        num_attempted += oie_dict_count(output_clauses)
        num_gold += oie_dict_count(target_clauses)
        num_correct += oie_dict_intersection(output_clauses, target_clauses)
    return prf1(num_correct, num_attempted, num_gold)


if __name__ == '__main__':
    s = '(Alice; has; the key) (Alice; sleep) (Bob; runs; fast; towards the school)'
    print(unserialize(s))