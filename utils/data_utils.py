import orjson
from typing import Optional, List, Dict, Union

import numpy as np
import torch
from dataclasses import dataclass
from transformers import BatchEncoding


def read_test_file(infile, n=None):
    answer = []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line != "":
                answer.append(line.split())
            if n is not None and len(answer) == n:
                break
    return answer


def load_ranking_dataset(infile):
    with open(infile, "rb") as fin:
        train_dataset, train_data = orjson.loads(fin.read())
    for elem in train_dataset:
        for key in ["hard_pairs", "soft_pairs", "no_change_pairs"]:
            elem[key] = np.array(elem[key]) if len(elem[key]) > 0 else np.zeros(shape=(0,2), dtype=int)
    return train_dataset, train_data


@dataclass
class LMDataCollator:
    pad_index: int = 0
    pad_to_multiple_of: Optional[int] = None
    padding_side: Optional[str] = "right"
    not_array_keys: Optional[List[str]] = None
    device: str = "cuda"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict:
        batch = dict()
        for key in features[0]:
            if self.not_array_keys is not None and key in self.not_array_keys:
                batch[key] = [elem[key] for elem in features]
            else:
                value = [np.array(elem[key]) for elem in features]
                max_length = max(len(x) for x in value)
                if self.pad_to_multiple_of is not None:
                    max_length += self.pad_to_multiple_of - 1
                    max_length -= max_length % self.pad_to_multiple_of
                for i, elem in enumerate(value):
                    diff = max_length - len(elem)
                    if diff > 0:
                        padding = np.tile(np.zeros_like(elem[:1]), diff)
                        to_concatenate = [padding, elem] if self.padding_side == "left" else [elem, padding]
                        value[i] = np.concatenate(to_concatenate, axis=0)
                batch[key] = torch.as_tensor(value, device=self.device)
        # batch = BatchEncoding(batch, tensor_type="pt")
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch