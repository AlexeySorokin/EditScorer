import itertools
from itertools import chain

import jsonlines
import os
import torch
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer
from sacremoses import MosesDetokenizer

from utils.tokenize_utils import tokenize_words_new
from utils.utils import do_spans_overlap, apply_simple_edit


class DatasetPreparer:
    
    def __init__(self, model="roberta-base", language=None, only_generated=False,
                 use_default=True, wrap_empty_edits=False, use_sep_for_default=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, add_prefix_space=True, strip_accents=False)
        self.detokenizer = MosesDetokenizer(lang=language) if language is not None else None
        self.tokenization_memo = dict()
        self.only_generated = only_generated
        self.use_default = use_default
        self.wrap_empty_edits = wrap_empty_edits
        self.use_sep_for_default = use_sep_for_default
    
    def detokenize(self, words):
        return self.detokenizer.detokenize(words.split())
    
    def words_to_input_ids(self, words):
        answer = self.tokenization_memo.get(words)
        if answer is None:
            if self.detokenizer is not None:
                words = self.detokenizer.detokenize(words.split())
            self.tokenization_memo[words] = self.tokenizer(words, add_special_tokens=False)["input_ids"]
            answer = self.tokenization_memo[words]
        return answer
    
    def _prepare_sample(self, sent, has_answers=True):
        data, default_index = [], None
        # input_ids_by_words = tokenize_words(sent["words"], self.tokenizer, self.detokenizer)
        input_ids_by_words = tokenize_words_new(sent["words"], self.tokenizer, detokenizer=self.detokenizer)
        if input_ids_by_words is None:
            return {"data": None}
        # input_ids_by_words = tokenize_words(sent["words"], self.tokenizer, self.detokenizer)
        word_offsets = [0] + list(np.cumsum([len(x) for x in input_ids_by_words]))
        input_ids = list(chain.from_iterable(input_ids_by_words))
        sent["edits"] = [edit for edit in sent["edits"] if edit["end"] <= len(input_ids_by_words)]
        sent["edits"] = [edit for edit in sent["edits"] if edit["start"] <= edit["end"]]
        for i, edit in enumerate(sent["edits"]):
            if edit["start"] >= 0:
                input_ids_to_insert = self.words_to_input_ids(edit["target"])
                # new_input_ids = input_ids[:edit["start"]] + input_ids_to_insert + input_ids[edit["end"]:]
                new_input_ids = list(chain.from_iterable(
                    input_ids_by_words[:edit["start"]] + [input_ids_to_insert] + input_ids_by_words[edit["end"]:]
                ))
                length_diff = len(new_input_ids) - len(input_ids)
                word_start, word_end = edit["start"], edit["end"]
                if word_end < word_start:
                    continue
                assert word_end <= len(word_offsets)
                if self.wrap_empty_edits and (word_end == word_start or len(input_ids_to_insert) == 0):
                        # wrapping insertion or deletion
                        word_start = max(word_start-1, 0)
                        word_end = min(word_end+1, len(input_ids_by_words))
                origin_start, origin_end = word_offsets[word_start] + 1, word_offsets[word_end] + 1
                if origin_end == origin_start:
                    origin_end += 1
                edit_start = origin_start + len(input_ids) + 1
                edit_end = origin_end + len(new_input_ids) + 1
                if edit_end <= edit_start:
                    edit_end = edit_start + 1
            else:
                if not self.use_default:
                    continue
                default_index = len(data)
                new_input_ids, origin_start, origin_end = input_ids[:], 0, 1
                if self.use_sep_for_default:
                    edit_start, edit_end = len(input_ids) + 1, len(input_ids) + 2
                else:
                    edit_start, edit_end = 0, 1
            pair_input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id] + new_input_ids
            if self.tokenizer.eos_token_id is not None:
                pair_input_ids.append(self.tokenizer.eos_token_id)
            else:
                pair_input_ids.append(self.tokenizer.sep_token_id)
            if max(edit_start, origin_start) >= len(pair_input_ids):
                # that check is expected to be always true, but we prevent bound violation
                continue
            assert origin_end >= origin_start
            output_edit = {
                "input_ids": pair_input_ids, "start": edit_start, "end": edit_end,
                "origin_start": origin_start, "origin_end": origin_end
            }
            if has_answers and "is_correct" in edit:
                output_edit["label"] = int(edit["is_correct"])
            data.append(output_edit)
        answer = {"data": data, "default": default_index}
        if has_answers:
            positive, negative, hard_pairs, soft_pairs, no_change_pairs, = [], [], [], [], []
            for i, elem in enumerate(data):
                (positive if elem["label"] else negative).append((i, elem))
                if default_index is not None and i != default_index:
                    no_change_pairs.append([i, default_index] if elem["label"] else [default_index, i])
            for (i_pos, elem_pos), (i_neg, elem_neg) in itertools.product(positive, negative):
                is_pair_hard = do_spans_overlap((elem_pos["start"], elem_pos["end"]), (elem_neg["start"], elem_neg["end"]))
                (hard_pairs if is_pair_hard else soft_pairs).append([i_pos, i_neg])
            hard_pairs = np.array(hard_pairs) if len(hard_pairs) > 0 else np.zeros(shape=(0, 2), dtype=int)
            soft_pairs = np.array(soft_pairs) if len(soft_pairs) > 0 else np.zeros(shape=(0, 2), dtype=int)
            no_change_pairs = np.array(no_change_pairs) if len(no_change_pairs) > 0 else np.zeros(shape=(0, 2), dtype=int)
            answer.update({"hard_pairs": hard_pairs, "soft_pairs": soft_pairs, "no_change_pairs": no_change_pairs})
        return answer
    
    def prepare(self, data, has_answers=True, n=None):
        if n is not None:
            data = data[:n]
        answer = [self._prepare_sample(elem, has_answers=has_answers) for elem in tqdm(data)]
        for i, elem in enumerate(answer):
            elem["indexes"] = i
        return answer
    

class BatchIndexesSampler:
    
    def __init__(self, data, bucket_size=100, total_batch_size=1500, random_seed=117):
        self.data = data
        self.lengths = [max(len(x["input_ids"]) for x in elem["data"]) for elem in self.data]
        self.bucket_size = bucket_size
        self.total_batch_size = total_batch_size
        self.random_seed = random_seed
        
    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.random_seed)
        indexes = torch.randperm(len(self.data), generator=generator).tolist()
        # indexes = torch.randperm(len(self.data)).tolist()
        buckets = [indexes[start:start+self.bucket_size] for start in range(0, len(self.data), self.bucket_size)]
        buckets = [sorted(bucket, key=lambda i: self.lengths[i], reverse=True) for bucket in buckets]
        for bucket in buckets:
            curr_indexes, curr_batch_size, batch_size = [], 0, None
            for index in bucket:
                if batch_size is None:
                    batch_size = self.total_batch_size // self.lengths[index]
                    curr_indexes, curr_batch_size = [], 0
                curr_indexes.append(index)
                curr_batch_size += len(self.data[index]["data"])
                if curr_batch_size >= batch_size:
                    batch_size = None
                    yield curr_indexes
            if len(curr_indexes) > 0:
                yield curr_indexes

class BatchCollator:

    def __init__(self, pad_fields=None, pad=0, tensor_fields=None, array_fields=None, other_fields=None, device="cuda"):
        self.pad_fields = pad_fields or ["input_ids"]
        self.tensor_fields = tensor_fields or ["input_ids", "label"]
        self.array_fields = array_fields or ["hard_pairs", "soft_pairs", "no_change_pairs"]
        self.other_fields = other_fields or ["indexes", "default"]
        self.pad = pad
        self.device = device
        
    def __call__(self, data):
        answer = {key: [x[key] for elem in data for x in elem["data"]] for key in data[0]["data"][0]}
        offsets = [0] + list(np.cumsum([len(x["data"]) for x in data]))
        for key in self.array_fields:
            if key in data[0]:
                answer[key] = np.concatenate([offset + elem[key] for offset, elem in zip(offsets, data)], axis=0)
        for key in self.other_fields:
            if key in data[0]:
                answer[key] = [elem[key] for elem in data]
        for key in self.pad_fields:
            L = max(len(x) for x in answer[key])
            for elem in answer[key]:
                elem.extend([self.pad] * (L-len(elem)))
        for key in self.tensor_fields:
            if key in answer:
                answer[key] = torch.as_tensor(answer[key], device=self.device)
        answer["offset"] = offsets
        return answer
    
    
def prepare_dataset(infiles, model="roberta-base", language="en",
                    use_default=True, only_with_positive=False,
                    wrap_empty_edits=False, use_sep_for_default=False,
                    has_answers=True, only_generated=True, min_diff=None,
                    n=None, return_flat=True):
    data = []
    if isinstance(infiles, str):
        infiles = [infiles]
    for infile in infiles:
        with jsonlines.open(infile) as fin:
            data += list(fin)
        if n is not None and len(data) >= n:
            data = data[:n]
            break
    for elem in chain.from_iterable(data):
        if isinstance(elem["words"], str):
            elem["words"] = elem["words"].split()
    if min_diff is not None:
        for elem in chain.from_iterable(data):
            elem["edits"] = [x for x in elem["edits"]
                             if ("diff" not in x or x.get("is_correct", False) or x["diff"] >= min_diff)]
            for edit in elem["edits"]:
                if "words" not in edit:
                    edit["words"] = apply_simple_edit(elem["words"], edit["start"], edit["end"], edit["target"])
    if not use_default:
        for elem in chain.from_iterable(data):
            elem["edits"] = [x for x in elem["edits"] if x["start"] >= 0]
    if only_with_positive:
        good_indexes = [i for i, elem in enumerate(data)
                        if any(edit["is_correct"] and edit["start"] >= 0 for sent in elem for edit in sent["edits"])]
        print(good_indexes[:10])
        data = [data[i] for i in good_indexes]
    if only_generated:
        # print(only_generated)
        for elem in chain.from_iterable(data):
            elem["edits"] = [x for x in elem["edits"] if x["is_generated"]]
    flat_data = list(chain.from_iterable(data))
    answer = DatasetPreparer(
        model=model, language=language, only_generated=only_generated,
        use_default=use_default, wrap_empty_edits=wrap_empty_edits,
        use_sep_for_default=use_sep_for_default
    ).prepare(flat_data, has_answers=has_answers)
    return answer, (flat_data if return_flat else data)

def prepare_dataloader(data, batch_size=1500, device="cuda"):
    batch_sampler = BatchIndexesSampler(data, total_batch_size=batch_size)
    return DataLoader(data, batch_sampler=batch_sampler, collate_fn=BatchCollator(device=device))


def output_predictions(predictions, data, file=None):
    fout = open(file, "w", encoding="utf8") if file is not None else None
    for curr_answer, curr_data in zip(predictions, data):
        print(*(curr_data["words"]), file=fout)
        prev_label = 1
        print("-" * 40, file=fout)
        for index in np.argsort(curr_answer["probs"])[::-1]:
            label, prob = curr_answer["labels"][index], curr_answer["probs"][index]
            if label < prev_label:
                print("-" * 40, file=fout)
            edit = curr_data["edits"][index]
            is_ok = "OK" if (edit["is_correct"] == bool(label)) else "ERROR"
            target = "" if edit["target"] is None else edit["target"]
            print(*(edit["words"]), file=fout)
            print(f"{edit['start']} {edit['end']} {edit['source'].replace(' ', '_')}->{target.replace(' ', '_')}", end=" ", file=fout)
            if "diff" in edit:
                print(f"diff={edit['diff']:.2f} score={prob:.2f} label={label} {is_ok}", file=fout)
            else:
                print(f"score={prob:.2f} label={label} {is_ok}", file=fout)
            prev_label = label
        if prev_label == 1:
            print("-" * 40, file=fout)
        print("", file=fout)