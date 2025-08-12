import json
import logging
import re
import sys

from datetime import datetime
from os.path import join as join_path, isfile as exist_file
from pprint import pp
from tqdm import tqdm
from typing import Any

logging.basicConfig(
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level = logging.DEBUG,
    stream = sys.stdout
)


def clean_split(s: str) -> list[str]:
    return [k.strip(' ') for k in s.strip(' ').strip(')').strip('(').strip(' ').split(') (')]


def multiset_intersection(a, b):
    """The key types should be hashable of course, and the values numerics"""
    total = 0
    for key in a.keys() & b.keys():
        total += min(a[key], b[key])
    return total


def list_hamming_dist(a: list[Any], b: list[Any]) -> int:
    """Count number of index matches"""
    return sum([1 if a[i] == b[i] else 0 for i in range(min(len(a), len(b)))])


def prf1(num_correct: int, num_attempted: int, num_gold: int) -> tuple[float, float, float]:
    precision = num_correct / num_attempted
    recall = num_correct / num_gold
    
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
        
    return precision, recall, f1


def logger(name: str = None):
    l = logging.getLogger(name)
    def passthru(self, v, n: str):
        self.debug(f"Assigned {n} to {v}")
        return v
    logging.Logger.passthru = passthru
    return l


def oie_part_counts(filename: str):
    counts = [0] * 8
    with open(filename, 'r') as f:
        for line in f:
            a = json.loads(line)['target']
            for b in re.finditer(r'\((.+?)\)', a):
                l = len(b.group(1).split(';')) - 1
                counts[l] += 1
    pp(counts)
    
    
def progress(iterable, desc: str):
    """Progress Bar for an Iterable Object"""
    return tqdm(iterable, ascii = True, desc = desc)


def getstr_safe(l: list[str], i: int):
    return l[i] if 0 <= i < len(l) else ''


def text_to_triple(outputs, targets):
    output_list = []
    target_list = []
    for output, target in zip(outputs, targets):

        sentence_outputs = []
        sentence_targets = []

        if output == "":
            pass
        else:
            _os = ['(' + k.strip(')').strip('(') + ')' for k in output.split(') (')]
            for o in _os:
                output_split = o.split(';')
                if len(output_split) < 3:
                    output_split.extend(["", "", ""])
                head, pred, tail = output_split[:3]
                sentence_outputs.append({
                    "head": head.replace("(", "").strip(' '),
                    "predicate": pred.strip(' '),
                    "tail": tail.replace(")", "").strip(' ')
                })

        if target == "":
            pass
        else:
            ts = ['(' + k.strip(' ').strip(')').strip('(') + ')' for k in target.split(') (')]
            for t in ts:
                target_split = t.split(';')
                if len(target_split) < 3:
                    target_split.extend(["", "", ""])
                head, pred, tail = target_split[:3]
                sentence_targets.append({
                    "head": head.replace("(", "").strip(' '),
                    "predicate": pred.strip(' '),
                    "tail": tail.replace(")", "").strip(' ')
                })

        output_list.append(sentence_outputs)
        target_list.append(sentence_targets)

    return output_list, target_list


def now() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


if __name__ == '__main__':
    oie_part_counts(sys.argv[1])