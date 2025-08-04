import logging
import sys

from pprint import pp
from typing import Any

logging.basicConfig(
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level = logging.DEBUG,
    stream = sys.stdout
)


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
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def logger(name: str = None) -> logging.Logger:
    return logging.getLogger(name or __name__)