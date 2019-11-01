import numpy as np


def print_seq_text(wf, codec):
    prev = 0
    word = ''
    current_word = ''
    start_pos = 0
    end_pos = 0
    dec_splits = []
    splits = []
    has_letter = False
    for cx in range(0, wf.shape[0]):
        c = wf[cx]
        if prev == c:
            if c > 2:
                end_pos = cx
            continue
        if 3 < c < (len(codec) + 4):
            ordv = codec[c - 4]
            char = ordv
            if char == ' ' or char == '.' or char == ',' or char == ':':
                if has_letter:
                    if char != ' ':
                        current_word += char
                    splits.append(current_word)
                    dec_splits.append(cx + 1)
                    word += char
                    current_word = ''
            else:
                has_letter = True
                word += char
                current_word += char
            end_pos = cx
        elif c > 0:
            if has_letter:
                dec_splits.append(cx + 1)
                word += ' '
                end_pos = cx
                splits.append(current_word)
                current_word = ''

        if len(word) == 0:
            start_pos = cx
        prev = c

    dec_splits.append(end_pos + 1)
    conf2 = [start_pos, end_pos + 1]

    return word.strip(), np.array([conf2]), np.array([dec_splits]), splits
