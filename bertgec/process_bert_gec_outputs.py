import bisect
import json
from argparse import ArgumentParser
from itertools import chain

import jsonlines
import re

from tqdm.auto import tqdm

from errant_wrapper.read_errant import read_m2_simple
from utils.data_utils import read_test_file
from utils.utils import find_single_edit, apply_simple_edit


def annotate(data, source_data, join_sentences=False, has_answers=True):
    UNK = re.escape("<unk> ")
    j, pos, last_match = 0, 0, -1
    matched_spans, sentence_offsets, block_offsets = set(), [0], [0]
    matched_sentence_indexes, curr_matched_sentence_indexes = [], []
    for i, sent_data in tqdm(enumerate(data)):
        words = sent_data["words"]
        pattern = (re.escape(" ".join(words)) + "\\ ").replace(UNK, "\s*.\s*")
        if pattern.endswith("\\ "):
            pattern = pattern[:-2]
        while True:
            if j >= len(source_data):
                # raise ValueError(f"no match for sentence {i}, {' '.join(words)}")
                print(f"no match for sentence {i}, {' '.join(words)}")
                j, pos = last_match + 1, 0
                curr_matched_sentence_indexes, matched_spans, sentence_offsets = [], set(), [0]
                break
            source_words = " ".join(source_data[j]["source"])
            match = re.search(pattern, source_words[pos:])
            if match is None:
                if pos != 0:
                    print(f"Error in sentence {i} with respect to {j}.")
                    pos, curr_matched_sentence_indexes, matched_spans = 0, [], set()
                    sentence_offsets = [0]
                    break
                else:
                    j += 1
            else:
                curr_matched_sentence_indexes.append(i)
                matched_words = source_words[pos+match.start():pos+match.end()].split()
                offset = sentence_offsets[-1]
                if has_answers:
                    correct_spans = {(edit.start, edit.end, edit.candidate) for edit in source_data[j]["edits"]}
                if words != matched_words:
                    change_start, change_end, _, change_target = find_single_edit(words, matched_words)
                    length_diff = len(change_target) - change_end + change_start
                else:
                    change_start, change_end, length_diff = None, None, 0
                for curr_edit in sent_data["edits"]:
                    if curr_edit["start"] < 0:
                        continue
                    start = curr_edit["start"] + offset
                    end = curr_edit["end"] + offset
                    target = curr_edit["target"]
                    if change_start is not None:
                        if start >= change_end:
                            start += length_diff
                            end += length_diff
                        else:
                            continue
                    curr_span = (start, end, target)
                    if has_answers:
                        if curr_span in matched_spans:
                            continue
                        if curr_span in correct_spans:
                            curr_edit["is_correct"] = True
                            matched_spans.add(curr_span)
                        elif end - start == len(target.split()) and all(
                                (r, r+1, word) in correct_spans for r, word in enumerate(target.split(), start)
                                ):
                            curr_edit["is_correct"] = True
                        else:
                            curr_edit["is_correct"] = False
                    curr_edit["is_generated"] = True
                    if change_start is not None and start >= change_end:
                        curr_edit["start"] += length_diff
                        curr_edit["end"] += length_diff
                pos += match.end()
                sent_data["words"] = matched_words
                sentence_offsets.append(sentence_offsets[-1] + len(matched_words))
                if has_answers:
                    has_corrections = False
                    for edit in source_data[j]["edits"]:
                        if edit.start < 0:
                            continue
                        if edit.start < sentence_offsets[-2] or edit.start >= sentence_offsets[-1]:
                            continue
                        has_corrections = True
                        if (edit.start, edit.end, edit.candidate) in matched_spans:
                            continue
                        start = int(edit.start) - sentence_offsets[-2]
                        end = int(edit.end) - sentence_offsets[-2]
                        curr_edit = {
                            "start": start, "end": end,
                            "target": edit.candidate,
                            "source": " ".join(source_data[j]["source"][edit.start:edit.end]),
                            "words": apply_simple_edit(matched_words, start, end, edit.candidate),
                            "diff": None,
                            "is_correct": True,
                            "is_generated": False
                        }
                        sent_data["edits"].append(curr_edit)
                curr_edit = {
                    "start": -1, "end": -1,
                    "target": None,
                    "source": "",
                    "words": matched_words,
                    "diff": 0.0,
                    "is_generated": True
                }
                if has_answers:
                    curr_edit["is_correct"] = not has_corrections
                sent_data["edits"].append(curr_edit)
                if pos == len(source_words):
                    matched_sentence_indexes.append(curr_matched_sentence_indexes)
                    curr_matched_sentence_indexes = []
                    last_match = j
                    pos, j = 0, j+1
                    matched_spans = set()
                    sentence_offsets = [0]
                    block_offsets.append(len(matched_sentence_indexes))
                break
    for sent_data in data:
        sent_data["edits"] = [elem for elem in sent_data["edits"] if "is_generated" in elem]
    if has_answers:
        for sent_data in data:
            sent_data["edits"] = [elem for elem in sent_data["edits"] if "is_correct" in elem]
    if join_sentences:
        annotated = [
            [data[i] for i in  elem] for elem in matched_sentence_indexes
        ]
    else:
        annotated = [[data[i]] for i in chain.from_iterable(matched_sentence_indexes)]
    return annotated
    
argument_parser = ArgumentParser()
argument_parser.add_argument("-i", "--infile", required=True)
argument_parser.add_argument("-s", "--source_file", required=True)
argument_parser.add_argument("-o", "--outfile", required=True)
argument_parser.add_argument("-j", "--join_sentences", action="store_true")
argument_parser.add_argument("-t", "--threshold", default=-2.0, type=float)
argument_parser.add_argument("-a", "--annotator_index", default=0, type=int)
argument_parser.add_argument("-r", "--raw", action="store_true")

if __name__ == "__main__":
    args = argument_parser.parse_args()
    if args.raw:
        source_data = [{"source": sent} for sent in read_test_file(args.source_file)]
    else:
        source_data = read_m2_simple(args.source_file)
        for elem in source_data:
            elem["edits"] = elem["edits"][args.annotator_index if (args.annotator_index < len(elem)) else 0]
    with jsonlines.open(args.infile, "r") as fin:
        input_data = list(fin)
    annotated = annotate(input_data, source_data, join_sentences=args.join_sentences, has_answers=not args.raw)
    if not args.raw:
        for block in annotated:
            for sent in block:
                sent["edits"] = [x for x in sent["edits"] if not x["is_generated"] or x["diff"] >= args.threshold]
        assert all(len(sent["edits"]) > 0 for block in annotated for sent in block)
        total_matched = sum(edit["is_correct"] and edit["is_generated"] and edit["start"] >= 0
                            for sent_data in chain.from_iterable(annotated)
                            for edit in sent_data["edits"])
        total = sum(edit["is_correct"] and edit["start"] >= 0
                    for sent_data in chain.from_iterable(annotated)
                    for edit in sent_data["edits"])
        print(f"{len(annotated)} {total_matched} {total} {(100 * total_matched / total):.2f}")
    else:
        print(f"{len(annotated)} {len(source_data)}")
    with open(args.outfile, "w", encoding="utf8") as fout:
        for elem in annotated:
            print(json.dumps(elem), file=fout)
        