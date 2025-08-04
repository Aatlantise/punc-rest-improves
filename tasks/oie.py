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
    

def unserialize(example: str, strict: bool = True) -> dict[str, dict[str, set[str]]]:
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
        components = [comp.strip(' )') for comp in clause.group(1).split(';')]
        if len(components) < 2:
            continue
        
        subject = components[0].strip()
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
        logger.debug('Text %s', text)
        output_clauses, target_clauses = unserialize(output, strict), unserialize(target, strict)
        
        logger.debug('Output %s', output)
        logger.debug('Output Dictionary')
        logger.debug(output_clauses)
        
        logger.debug('Target %s', target)
        logger.debug('Target Dictionary')
        logger.debug(target_clauses)
        
        attempted, gold, correct = oie_dict_count(output_clauses), oie_dict_count(target_clauses), oie_dict_intersection(output_clauses, target_clauses)
        logger.debug('Attempted %d', attempted)
        logger.debug('Gold %d', gold)
        logger.debug('Correct %d', correct)
        
        num_attempted += attempted
        num_gold += gold
        num_correct += correct
        
        logger.debug('\n')
        
    return prf1(num_correct, num_attempted, num_gold)


if __name__ == '__main__':
    s = '(The second; titled;  Consider Her Ways '') (Her Ways; consider) (The second; starred; as the lead named Jane Waterleigh; Barrie) ( Consider Her Ways ''; starred; as the lead named Jane Waterleigh; Barrie) (the lead; named; Jane Waterleigh)'
    print(unserialize(s))