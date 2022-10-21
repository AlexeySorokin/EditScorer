import Levenshtein
import difflib


def extract_edits(first, second):
    first_text, second_text = " ".join(first), " ".join(second)
    first_ends = [0] + [i for i, letter in enumerate(first_text) if letter == " "] + [len(first_text)]
    second_ends = [0] + [i for i, letter in enumerate(second_text) if letter == " "] + [len(second_text)]
    first_ends = {pos: i for i, pos in enumerate(first_ends)}
    second_ends = {pos: i for i, pos in enumerate(second_ends)}
    opcodes = Levenshtein.opcodes(first_text, second_text)
    alignment = []
    for code, i_first, j_first, i_second, j_second in opcodes:
        if code == "insert":
            alignment.extend([(i_first, i) for i in range(i_second, j_second)])
        elif code == "delete":
            alignment.extend([(i, i_first) for i in range(i_first, j_first)])
        elif code == "equal":
            alignment.extend(zip(range(i_first, j_first), range(i_second, j_second)))
        else:
            alignment.append((i_first, i_second))
    alignment.append((len(first_text), len(second_text)))
    answer = []
    for i, j in alignment:
        first_pos, second_pos = first_ends.get(i), second_ends.get(j)
        if first_pos is not None and second_pos is not None:
            answer.append((first_pos, second_pos))
    indexes_to_keep, state = [0], None
    for i, (i_first, i_second) in enumerate(answer[1:], 1):
        prev_state = state
        if i_first == answer[i - 1][0]:
            state = "insert"
        elif i_second == answer[i - 1][1]:
            state = "delete"
        else:
            state = None
        if state != prev_state and prev_state is not None:
            indexes_to_keep.append(index)
        if state is not None:
            index = i
        else:
            indexes_to_keep.append(i)
    if state is not None:
        indexes_to_keep.append(index)
    answer = [answer[i] for i in indexes_to_keep]
    return [(i_first, answer[i + 1][0], i_second, answer[i + 1][1])
            for i, (i_first, i_second) in enumerate(answer[:-1])]


def align_sentences(first, second):
    answer = []
    opcodes = difflib.SequenceMatcher(None, first, second).get_opcodes()
    for code, i_first, j_first, i_second, j_second in opcodes:
        if code == "insert":
            answer.append((i_first, j_first, " ".join(second[i_second:j_second])))
        elif code == "delete":
            answer.append((i_first, j_first, ""))
        elif code == "replace":
            groups = extract_edits(first[i_first:j_first], second[i_second:j_second])
            for r_first, s_first, r_second, s_second in groups:
                answer.append(
                    (i_first + r_first, i_first + s_first, " ".join(second[i_second + r_second:i_second + s_second]))
                )
    return answer