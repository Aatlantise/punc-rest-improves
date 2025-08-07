from utils import list_hamming_dist, prf1, logger

logger = logger()


def score(texts: list[str], outputs: list[str], targets: list[str], strict = False) -> tuple[float, float, float]:
    """Score POS by matching"""
    num_correct, num_attempted, num_gold = 0, 0, 0
    for text, output, target in zip(texts, outputs, targets):
        output_tags, target_tags = output.split(), target.split()
        num_attempted += len(output_tags)
        num_gold += len(target_tags)
        num_correct += list_hamming_dist(output_tags, target_tags)
    return prf1(num_correct, num_attempted, num_gold)