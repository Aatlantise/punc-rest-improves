import re
from utils import progress

IN_FILENAME = 'wikipedia.txt'
OUT_FILENAME = 'wikipedia-processed.txt'


def bookending_chars(s: str) -> tuple[str, str]:
    stripped = s.strip()
    if len(stripped) < 1:
        return '', ''
    return stripped[0], stripped[-1]

def is_like_citation(s: str) -> bool:
    return re.match(r'\s[A-Za-z\d.\u2013]+', s) is not None

def is_like_sentence(s: str) -> bool:
    if len(s) <= 30:
        return False
    first, last = bookending_chars(s)
    return first.isalpha() and last in set(',.;:!?') and not is_like_citation(s)

def rid_empty_brackets(s: str) -> str:
    s = re.sub(r'\(\W*?\)', '', s)
    s = re.sub(r'\[\W*?\]', '', s)
    return s

def rid_multiple_spaces(s: str) -> str:
    s = re.sub(r' +', ' ', s)
    return s

def rid_repeating_punctuations(s: str) -> str:
    s = re.sub(r'([,;\.] ?){2,}', '\1', s)
    return s

def rid_brackets_with_missing_content(s: str) -> str:
    s = re.sub(r'\(\W+.+?\)', '', s)
    s = re.sub(r'\(.+?\W+\)', '', s)
    s = re.sub(r'\[\W+.+?\]', '', s)
    s = re.sub(r'\[.+?\W+\]', '', s)
    return s

def normalize_spaces(s: str) -> str:
    s = re.sub(r'\x01', ' ', s)
    s = re.sub(r'\u00a0', ' ', s)
    return s

def pre_process(s: str) -> str:
    s = rid_empty_brackets(s)
    s = rid_repeating_punctuations(s)
    s = rid_brackets_with_missing_content(s)
    return s

def post_process(s: str) -> str:
    s = rid_empty_brackets(s)
    s = rid_repeating_punctuations(s)
    s = normalize_spaces(s)
    s = rid_multiple_spaces(s)
    return s

def cleaned(lines: list[str]) -> list[str]:
    counter = 0
    picked_lines = []
    for long_lines in progress(lines, 'Picking Lines'):
        for l in long_lines.split('\n'):
            counter += 1
            if re.match(r'\s[,.;]\s', l):
                continue  # bad sentence, missing words
            elif is_like_sentence(l):
                picked_lines.append(l)
    print('Picked %d lines out of %d' % (len(lines), counter))
    counter = 0
    last_sentence = ''
    cleaned_lines = []
    for l in progress(picked_lines, 'Cleaning Lines'):
        l = pre_process(l)
        first, _ = bookending_chars(l)
        if first == '':
            continue
        elif last_sentence == '':
            last_sentence = l
        elif first.islower():
            last_sentence += ' ' + l.lstrip(' \t')
        elif l[-1] == ':' or ' . ' in l or ' : ' in l:  # incomplete list or citation-like
            continue
        else:
            final = post_process(last_sentence).strip()
            cleaned_lines.append(final + '\n')
            counter += 1
            last_sentence = l
    print('Wrote %s lines' % counter)
    return cleaned_lines


if __name__ == '__main__':
    counter = 0
    cycle = 0
    wrote_lines = 0
    lines = []
    with open(IN_FILENAME, 'r') as f_in:
        with open(OUT_FILENAME, 'w') as f_out:
            for i, line in enumerate(f_in):
                counter += 1
                lines.append(line)
                if counter > 50000:
                    cleaned_lines = cleaned(lines)
                    f_out.writelines(cleaned_lines)
                    wrote_lines += len(cleaned_lines)
                    lines = []
                    counter = 0
                    cycle += 1
                # if cycle > 20:
                #     break
                if wrote_lines > 450000:
                    break
            cleaned_lines = cleaned(lines)
            f_out.writelines(cleaned_lines)
            wrote_lines += len(cleaned_lines)
    print('Wrote %s lines in total' % wrote_lines)









