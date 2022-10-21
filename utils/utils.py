import numpy as np

def flatten(variants, stop_condition, filter_condition=None, offset=0):
    if not isinstance(variants, list):
        raise ValueError("function `_extract_texts_from_variants` works only with nested lists.")
    if len(variants) == 0:
        return [], []
    if not stop_condition(variants[0]):
        answer, arrangement = [], []
        for elem in variants:
            curr_answer, curr_arrangement = flatten(
                elem, stop_condition=stop_condition, filter_condition=filter_condition, offset=offset
            )
            answer += curr_answer
            arrangement.append(curr_arrangement)
            offset += len(curr_answer)
        return answer, arrangement
    if filter_condition is not None:
        variants = list(filter(filter_condition, variants))
    return variants, list(range(offset, offset+len(variants)))

def unflatten(variants, arrangements):
    if len(arrangements) == 0:
        return []
    if isinstance(arrangements[0], int):
        return [variants[i] for i in arrangements]
    return [unflatten(variants, elem) for elem in arrangements]

def find_common_prefix_length(first, second):
    if len(first) > len(second):
        first, second = second, first
    for i, (x, y) in enumerate(zip(first, second)):
        if x != y:
            return i
    return len(first)

def find_single_edit(first, second):
    if first == second:
        return None, [], []
    prefix_length = find_common_prefix_length(first, second)
    suffix_length = find_common_prefix_length(first[prefix_length:][::-1], second[prefix_length:][::-1])
    return (prefix_length,
            len(first)-suffix_length,
            first[prefix_length:len(first)-suffix_length],
            second[prefix_length:len(second)-suffix_length])

def apply_simple_edit(words, start, end, variant):
    if start < 0:
        return words
    left_context, right_context = words[:start], words[end:]
    if end == 0:
        right_context[0] = right_context[0].lower()
    variant_words = variant.split()
    if start == 0 and len(words) > 0 and words[0][0].isupper():
        if len(variant_words) > 0 and variant_words[0].islower():
            variant_words[0] = variant_words[0].title()
    new_words = left_context + variant_words + right_context
    return new_words

def annotate_variant(variant, edits):
    edit_triples = set()
    for edit in edits:
        if edit.start >= 0:
            edit_triples.add((edit.start, edit.end, edit.candidate))
        if variant["start"] == edit.start and variant["end"] == edit.end and variant["correction"] == edit.candidate:
            variant["is_correct"] = True
        elif variant["correction"] is None and edit.label == "noop":
            variant["is_correct"] = True
    if variant["end"] > variant["start"] + 1:
        if variant["correction"] == "":
            splitted_correction = [""] * (variant["end"] - variant["start"])
        else:
            splitted_correction = variant["correction"].split()
        if len(splitted_correction) == variant["end"] - variant["start"]:
            for start, word in enumerate(splitted_correction, variant["start"]):
                if (start, start+1, word) not in edit_triples:
                    break
            else:
                variant["is_correct"] = True
    if "is_correct" not in variant:
        variant["is_correct"] = False


def word_to_error_span(start, end):
    """
    :param start: int, the beginning of the span
    :param end: int, the end of the span
    :return:
        span_start: int, the beginning of the span in coverage notation
        span_end: int, the end of the span in coverage notation
        
    Returns span bounds in the sequence of elementary spans (0,0), (0,1), (1,1), ...
    Formally, maps a pair (s, t) to (2s+1, 2t) if s < t and maps (s, s) to (2s, 2s+1)
    """
    return (2 * start + int(end > start), 2*end+int(end == start))

def do_spans_overlap(first, second):
    i_first,  j_first = word_to_error_span(*first)
    i_second,  j_second = word_to_error_span(*second)
    return (i_first <= i_second < j_first) or (i_second <= i_first < j_second)


def make_offsets(lengths, add_last=True):
    answer = [0] + list(np.cumsum((lengths)))
    if not add_last:
        answer.pop()
    return answer