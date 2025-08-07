from utils import logger

logger = logger()


def process(tokens: list[str], tags: list[str]) -> tuple[str, str]:
    source = ' '.join(tokens)
    target = []
    current = ""
    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            # if it starts with B- then this is the start of new phrase
            if current:
                target.append(current)
            current = f"({token}:{tag[2:]})"
        elif tag.startswith("I-"):
            # if it starts with I- then word is inside a phrase
            # 2 for the I- and 1 for the ":" and 1 for the ")"
            current = current[:-len(tag[2:]) - 2] + f" {token}:{tag[2:]})"
        else:
            if current:
                target.append(current)
            current = ""
    if current:
        target.append(current)
    return source, ' '.join(target) if target else "O"  # O for no prediction