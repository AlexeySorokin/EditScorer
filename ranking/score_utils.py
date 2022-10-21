from copy import deepcopy
from itertools import chain

import numpy as np
import torch

from errant_wrapper.edit import apply_edits, Edit
from utils.generation_utils import update_words_and_variants, _finalize_correction
from ranking.data import DatasetPreparer, prepare_dataloader
from ranking.metrics import extract_labels, probs_to_labels
from utils.utils import apply_simple_edit, make_offsets


def extract_edits(data, only_generated=True, threshold=2.0, n_max=1, min_dist=5, key="diff", from_log=False):
    answer = []
    for elem in data:
        are_covered = np.zeros(shape=(len(elem["words"])), dtype=bool)
        curr_answer = []
        for edit in sorted(elem["edits"], key=lambda x: x[key], reverse=True):
            score = edit[key]
            if edit["start"] < 0:
                are_covered[:] = True
                break
            if score < threshold:
                break
            if only_generated and not edit["is_generated"]:
                continue
            if any(are_covered[i] for i in range(edit["start"], min(edit["end"] + 1, len(are_covered)))):
                continue
            curr_answer.append(edit)
            cover_start, cover_end = max(edit["start"] - min_dist, 0), min(edit["end"] + min_dist, len(elem["words"]))
            are_covered[cover_start:cover_end] = True
            if len(curr_answer) >= n_max:
                break
        answer.append(curr_answer)
    return answer

def print_corrections(source_data, answer, extracted, outfile=None):
    fout = open(outfile, "w", encoding="utf8") if outfile is not None else None
    for i, (curr_source_data, curr_extracted) in enumerate(zip(source_data, extracted)):
        source = curr_source_data['source']
        if isinstance(source, list):
            source = " ".join(source)
        print(f"{i + 1}\t{source}", file=fout)
        for j, curr_answer in enumerate(curr_extracted):
            if len(curr_answer) > 1:
                print(" ".join(answer[i][j]['words']), file=fout)
            for edit in curr_answer:
                if edit["start"] != edit["end"]:
                    source = " ".join(answer[i][j]["words"][edit["start"]:edit["end"]])
                else:
                    source = "_"
                target = "_" if edit["target"] == "" else edit["target"]
                status = "" if "is_correct" not in edit else 'OK' if edit['is_correct'] else 'ERROR'
                print(f"{edit['start']} {edit['end']} {source}->{target} {status}", file=fout)
                # edits.append(Edit(edit['start'], edit['end'], edit["target"], "-NONE-"))
        print("", file=fout)
    if fout is not None:
        fout.close()
    return

def make_output_data(source_data, extracted, sentence_offsets):
    output_data = []
    for i, (curr_source_data, curr_group) in enumerate(zip(source_data, extracted)):
        curr_edits = []
        for j, sent_corrections in enumerate(curr_group):
            for edit in sent_corrections:
                start, end = edit["start"] + sentence_offsets[i][j], edit["end"] + sentence_offsets[i][j]
                curr_edits.append(Edit(start, end, edit["target"], "-NONE-"))
        output_sent = apply_edits(curr_source_data["source"], curr_edits)
        output_data.append(output_sent["sent"])
    return output_data

def score_corrections(source_data, answer, threshold, key="diff", n_max=1, min_dist=5, outfile=None):
    extracted = [
        extract_edits(elem, only_generated=True, key=key, threshold=threshold, n_max=n_max, min_dist=min_dist)
        for elem in answer
    ]
    # calculating metrics
    TP = sum(int(edit["is_correct"]) for elem in chain.from_iterable(extracted) for edit in elem)
    FP = sum(len(elem) for elem in chain.from_iterable(extracted)) - TP
    FN = sum(int(edit.start >= 0) for elem in source_data for edit in elem["edits"]) - TP
    # printing output
    print_corrections(source_data, answer, extracted, outfile)
    # collecting edits and calculating output data
    sentence_offsets = [make_offsets([len(sent["words"]) for sent in elem]) for elem in answer]
    output_data = []
    for i, (curr_source_data, curr_group) in enumerate(zip(source_data, extracted)):
        curr_edits = []
        for j, sent_corrections in enumerate(curr_group):
            for edit in sent_corrections:
                start, end = edit["start"] + sentence_offsets[i][j], edit["end"] + sentence_offsets[i][j]
                curr_edits.append(Edit(start, end, edit["target"], "-NONE-"))
        output_sent = apply_edits(curr_source_data["source"], curr_edits)
        output_data.append(output_sent["edits"])
    return TP, FP, FN, output_data


def predict_by_stages(model, data, tokenizer="roberta-base", language=None,
                      batch_size=4000, rounds=3, threshold=0.5, n_max=3, min_dist=5,
                      alpha_source=0.0):
    # alpha_threshold = np.log(threshold) + alpha_source
    dataset_preparer = DatasetPreparer(model=tokenizer, language=language, only_generated=True)
    for elem in data:
        for edit in elem["edits"]:
            edit["actual_start"], edit["actual_end"] = edit["start"], edit["end"]
    curr_data = deepcopy(data)
    active_indexes = list(range(len(data)))
    answer = [[] for _ in data]
    for round in range(rounds):
        dataset = dataset_preparer.prepare(curr_data, has_answers=("label" in data[0]))
        short_dataset, long_indexes = [], []
        for i, elem in enumerate(dataset):
            if max(len(x["input_ids"]) for x in elem["data"]) <= model.max_length:
                short_dataset.append(elem)
            else:
                long_indexes.append(i)
        curr_answer = [None] * len(dataset)
        dataloader = prepare_dataloader(short_dataset, batch_size=batch_size, device="cuda")
        for batch in dataloader:
            model.eval()
            with torch.no_grad():
                batch_output = {"probs": model(**batch)}
            probs, batch_info = extract_labels(batch_output, batch)
            for curr_probs, elem in zip(probs, batch_info):
                curr_answer[elem["index"]] = {
                    "probs": curr_probs, "labels": probs_to_labels(curr_probs, elem["default"], threshold=0.5)
                }
        for i in long_indexes:
            elem = dataset[i]
            diffs = np.array([x.get("diff", 0.0) for x in elem["data"]])
            probs = 1 / (1.0 + np.exp(-diffs))
            curr_answer[i] = {"probs": probs, "labels": probs_to_labels(probs, elem["default"], threshold=threshold)}
        for curr_elem, curr_predictions in zip(curr_data, curr_answer):
            assert len(curr_elem["edits"]) == len(curr_predictions["probs"])
            for edit, prob in zip(curr_elem["edits"], curr_predictions["probs"]):
                edit["score"] = np.log(prob) + alpha_source * edit.get("diff", 0.0)
        extracted = extract_edits(curr_data, only_generated=True, key="score",
                                  threshold=threshold, n_max=n_max, min_dist=min_dist)
        new_active_indexes, new_data = [], []
        print(len(extracted), len(active_indexes))
        for i, curr_extracted in enumerate(extracted):
            new_variants, new_words = curr_data[i]["edits"], curr_data[i]["words"]
            for correction in sorted(curr_extracted, key=lambda x: x["end"], reverse=True):
                new_variants, new_words = update_words_and_variants(
                    correction, new_variants, new_words, target_key="target")
            if len(curr_extracted) > 0:
                index = active_indexes[i]
                for correction in curr_extracted:
                    correction = _finalize_correction(correction)
                    correction["source"] = data[index]["words"]
                    correction["words"] = apply_simple_edit(
                        data[index]["words"], correction["start"], correction["end"], correction["target"]
                    )
                    correction["stage"] = round+1
                    answer[index].append(correction)
                if len(new_words) > 0:
                    new_active_indexes.append(index)
                    new_data.append({"words": new_words, "edits": new_variants})
        active_indexes, curr_data = new_active_indexes, new_data
        if len(new_data) == 0:
            break
    return answer