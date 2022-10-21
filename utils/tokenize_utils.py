import sys

import numpy as np
from transformers import BertTokenizer, BertTokenizerFast, XLNetTokenizer, XLNetTokenizerFast


def to_title(s):
    if any(x.isupper() for x in s[1:]):
        return s
    return s.title()

def to_lower(s):
    if any(x.isupper() for x in s[1:]):
        return s
    return s.lower()

def is_alphabet_word(s, allow_hyphen=True):
    return all(x.isalpha() or x.isdigit() or x == "-" for x in s) and any(x.isalpha() or x.isdigit() for x in s)

def merge_sorted_unique(first, second):
    first_index, second_index = 0, 0
    answer = []
    while first_index < len(first) and second_index < len(second):
        first_value, second_value = first[first_index], second[second_index]
        if first_value <= second_value:
            answer.append(first_value)
            first_index += 1
        else:
            answer.append(second_value)
        if second_value <= first_value:
            second_index += 1
    return answer

memo = dict()

def tokenize_words(words, tokenizer, detokenizer, add_bos=False, add_eos=False):
    if isinstance(words, str):
        words = words.strip().split()
    if getattr(tokenizer, "do_lower_case", False):
        words = [word.lower() for word in words]
    text_words = detokenizer.detokenize(words).split()
    # text_words = detokenizer(words).split()
    assert "".join(words) == "".join(text_words), \
        f"Source words: {' '.join(words)}, after detokenization {' '.join(text_words)}"
    splits = [0] + merge_sorted_unique(np.cumsum([len(x) for x in text_words]),
                                       np.cumsum([len(x) for x in words]))
    joint_text = "".join(words).strip()
    text = [joint_text[start:end] for start, end in zip(splits[:-1], splits[1:])]
    token_ids = tokenizer(text, add_special_tokens=False, is_split_into_words=True)["input_ids"]
    # print(tokenizer.decode(token_ids).replace(" ", ""), "".join(text))
    # assert tokenizer._decode(token_ids).replace(" ", "") == "".join(text)
    # assert "".join(words) == "".join(tokens), str(words) + " != " + str(tokens)
    answer, curr_tokens, word_index, pos_in_word, token_index = [], [], 0, 0, 0
    while token_index < len(token_ids):
        next_token_index = token_index + 1
        while next_token_index <= len(token_ids):
            next_token = tokenizer._decode(token_ids[token_index:next_token_index]).strip()
            if isinstance(tokenizer, (BertTokenizer, BertTokenizerFast)):
                next_token = next_token.strip("#")
            # key = tuple(token_ids[token_index:next_token_index])
            # next_token = memo.get(key)
            # if next_token is None:
            #     next_token = tokenizer._decode(key).strip()
            #     memo[key] = next_token
            # print(token_index, word_index, next_token, words[word_index][pos_in_word:])
            if words[word_index][pos_in_word:].startswith(next_token):
                curr_tokens.extend(token_ids[token_index:next_token_index])
                token_index = next_token_index
                pos_in_word += len(next_token)
            elif next_token == tokenizer.unk_token:
                curr_tokens.extend(tokenizer.unk_token)
                token_index = next_token_index
                pos_in_word += 1
            if pos_in_word == len(words[word_index]):
                answer.append(curr_tokens)
                curr_tokens, pos_in_word = [], 0
                word_index += 1
                break
            next_token_index += 1
        else:
            raise ValueError(f"Bad token ids {token_ids} for words {words}")
    if add_bos:
        answer = [[tokenizer.bos_token_id]] + answer
    if add_eos:
        answer = answer + [[tokenizer.eos_token_id]]
    return answer


def find_offsets(text, tokens):
    end, answer = 0, []
    for token in tokens:
        start = text.find(token, end)
        if start < 0:
            return None
        answer.append(start)
        end = start + len(token)
    return answer

def find_missing_starts(starts, spans):
    positions_for_spaces = []
    pos_in_words = 0
    for pos_in_tokens, (token_start, token_end) in enumerate(spans):
        if token_start == starts[pos_in_words]:
            pos_in_words += 1
        while pos_in_words < len(starts) and token_end > starts[pos_in_words]:
            positions_for_spaces.append(starts[pos_in_words])
            pos_in_words += 1
        if pos_in_words >= len(starts):
            break
    return positions_for_spaces

def find_uncovered_starts(starts, spans):
    pos_in_words, answer = 0, []
    for pos_in_tokens, (token_start, token_end) in enumerate(spans):
        while pos_in_words < len(starts) and starts[pos_in_words] < token_end:
            if starts[pos_in_words] < token_start:
                answer.append(starts[pos_in_words])
            pos_in_words += 1
        if pos_in_words >= len(starts):
            break
    answer.extend(starts[pos_in_words:])
    return answer

def tokenize_words_new(words, tokenizer, detokenizer=None, add_bos=False, add_eos=False, max_rounds=3):
    if isinstance(words, str):
        words = words.strip().split()
    if getattr(tokenizer, "do_lower_case", False):
        words = [word.lower() for word in words]
    if isinstance(tokenizer, (XLNetTokenizer, XLNetTokenizerFast)):
        words = [word.replace("''", "\"") for word in words]
    if detokenizer is not None:
        text = detokenizer.detokenize(words)
    else:
        text = " ".join(words)
    word_offsets = find_offsets(text, words)
    if word_offsets is None:
        text = " ".join(words)
        word_offsets = find_offsets(text, words)
    def find_and_check_tokenization():
        tokenization = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offset_mapping = list(map(list, tokenization["offset_mapping"]))
        for i, (start, end) in enumerate(offset_mapping):
            while start < end and text[start] == " ":
                start += 1
            offset_mapping[i][0] = start
        if len(offset_mapping) == 0:
            return []
        if offset_mapping[0][0] > 1:
            print(text)
            print(word_offsets)
            print(list(zip(offset_mapping, (tokenizer._decode([x]) for x in tokenization["input_ids"]))))
            return None, None
        offset_mapping[0][0] = 0
        if find_uncovered_starts(word_offsets, offset_mapping):
            print(text)
            print(word_offsets)
            print(list(zip(offset_mapping, (tokenizer._decode([x]) for x in tokenization["input_ids"]))))
            return None, None
        return tokenization["input_ids"], offset_mapping
    input_ids, offset_mapping = find_and_check_tokenization()
    if offset_mapping is None:
        return None
    # проверяем, что никакие сабтокены не пересекают границы слов
    positions_for_spaces = find_missing_starts(word_offsets, offset_mapping)
    rounds = 0
    while len(positions_for_spaces) > 0:
        if rounds >= max_rounds:
            return None
        for pos in positions_for_spaces[::-1]:
            text = text[:pos] + " " + text[pos:]
        word_offsets = find_offsets(text, words)
        input_ids, offset_mapping = find_and_check_tokenization()
        if offset_mapping is None:
            return None
        positions_for_spaces = find_missing_starts(word_offsets, offset_mapping)
        rounds += 1
    pos_in_words, word_bounds = 0, []
    for pos_in_tokens, (token_start, token_end) in enumerate(offset_mapping):
        if pos_in_tokens == 0 and token_start <=1:
            token_start = 0
        if pos_in_words >= len(word_offsets):
            break
        if token_start == word_offsets[pos_in_words]:
            word_bounds.append(pos_in_tokens)
            pos_in_words += 1
    word_bounds.append(len(offset_mapping))
    answer = [input_ids[start:end] for start, end in zip(word_bounds[:-1], word_bounds[1:])]
    return answer

def tokenize_with_space(s, tokenizer, detokenizer, prefix="a"):
    text = detokenizer.detokenize([prefix, s])[len(prefix):]
    return tokenizer(text)["input_ids"]


class MemoTokenizer:
    
    def __init__(self, tokenizer, detokenizer, prefix="a"):
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer
        self.prefix = prefix
        self.memo = dict()
        
    def __call__(self, s):
        if s not in self.memo:
            self.memo[s] = self.tokenizer(self.detokenizer.detokenize([self.prefix, s])[len(self.prefix):])
        return self.memo[s]