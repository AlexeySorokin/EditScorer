import os
import re
import sys
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from tqdm.auto import tqdm


def attach_index(path, index, suffix=""):
    if re.search(suffix + "$", path):
        prefix, suffix = re.match(f"^(.*)({suffix})$", path).groups()
    else:
        prefix, suffix = path, ""
    return f"{prefix}_{index}{suffix}"

def get_batch_metrics(pred_labels, labels, mask=None, ignore_labels=None, metric_func=None):
    answer = defaultdict(int)
    for r, (curr_pred_labels, curr_labels) in enumerate(zip(pred_labels, labels)):
        if mask is not None:
            curr_labels = [x for x, flag in zip(curr_labels, mask[r]) if flag]
        elif ignore_labels is not None:
            curr_labels = [label for label in curr_labels if label not in ignore_labels]
        # assert len(curr_pred_labels) == len(curr_labels), f"{len(curr_pred_labels)}-{len(curr_labels)}"
        for key, value in metric_func(curr_labels, curr_pred_labels).items():
            answer[key] += value
    return answer

def update_metrics(metrics, batch_output, batch, mask=None,
                   answer_field="labels", y_field="y", extract_func=None,
                   metric_func=None, aggregate_func=None):
    n_batches = metrics["n_batches"]
    for key, value in batch_output.items():
        if "loss" in key:
            metrics[key] = (metrics.get(key, 0.0) * n_batches + value.item()) / (n_batches + 1)
    metrics["n_batches"] += 1
    if extract_func is not None:
        y_pred, y_true = extract_func(batch_output, batch)
    else:
        y_pred, y_true = batch_output[answer_field], batch[y_field].cpu().tolist()
    batch_metrics = get_batch_metrics(y_pred, y_true, mask=mask, ignore_labels=None, metric_func=metric_func)
    for key, value in batch_metrics.items():
        metrics[key] = metrics.get(key, 0) + value
    # print(metrics)
    aggregate_func(metrics)
    return batch_metrics



class ModelTrainer:
    
    def __init__(self, epochs=1, initial_epoch=0,
                 checkpoint_dir=None, checkpoint_name="checkpoint.pt", save_all_checkpoints=False,
                 eval_steps=None, evaluate_after=False, validate_metric="accuracy", less_is_better=False):
        self.epochs = epochs
        self.initial_epoch = initial_epoch
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        else:
            self.checkpoint_path = None
        self.save_all_checkpoints = save_all_checkpoints
        self.eval_steps = eval_steps
        self.evaluate_after = evaluate_after
        self.validate_metric = validate_metric
        self.less_is_better = less_is_better

    def do_epoch(self, model, dataloader, mode="validate", epoch=0, eval_steps=None,
                 answer_field="labels", y_field="y",
                 extract_func=None, metric_func=None, aggregate_func=None, display_func=None,
                 ncols=200, dynamic_ncols=False, count_mode="batch", total=None,
                 check_field="input_ids", check_dim=1, max_length=512):
        metrics = {"n_batches": 0, "loss": 0.0}
        func = model.train_on_batch if mode == "train" else model.validate_on_batch
        if count_mode == "batch":
            total = getattr(dataloader, "__len__", None)
        progress_bar = tqdm(total=total, leave=True, ncols=ncols, dynamic_ncols=dynamic_ncols)
        progress_bar.set_description(f"{mode}, epoch={(epoch + 1) if mode=='train' else epoch}")
        evaluation_step = 0
        with progress_bar:
            for batch in dataloader:
                prev_evaluation_step = evaluation_step
                if (mode == "train" and check_field in batch and batch[check_field].shape[check_dim] > max_length):
                    batch_metrics = dict()
                else:
                    batch_answers, mask = batch[y_field], batch.get("mask")
                    if mask is not None:
                        mask = mask.bool()
                    try:
                        if progress_bar.n <= -1 and mode == "train":
                            batch_metrics = dict()
                        else:
                            batch_output = func(batch, mask=mask)
                            batch_metrics = update_metrics(
                                metrics, batch_output, batch, mask, answer_field=answer_field, y_field=y_field,
                                extract_func=extract_func, metric_func=metric_func, aggregate_func=aggregate_func
                            )
                    except ValueError:
                        continue
                postfix = display_func(metrics)
                if count_mode == "sample":
                    batch_size = batch_metrics["seq_total"] if "seq_total" in batch_metrics else len(batch['indexes'])
                progress_bar.update(batch_size if count_mode == "sample" else 1)
                postfix["lr"] = f"{model.scheduler.get_last_lr()[0]:.2e}"
                progress_bar.set_postfix(postfix)
                if mode == "train" and eval_steps is not None:
                    evaluation_step = progress_bar.n // eval_steps
                    if evaluation_step != prev_evaluation_step:
                        self.eval_func(model, epoch=f"{epoch}_{progress_bar.n}")
        return metrics
    
    def train(self, model, train_data, dev_data=None, total=None, dev_total=None, count_mode="sample", **kwargs):
        self.best_score = np.inf if self.less_is_better else -np.inf
        eval_steps = self.eval_steps if dev_data is not None else None
        self.eval_func = partial(
            self.evaluate_and_save_model, dev_data=dev_data, total=dev_total,  count_mode=count_mode, **kwargs
        )
        for epoch in range(self.initial_epoch, self.epochs):
            train_metrics = self.do_epoch(
                model, train_data, mode="train", epoch=epoch, total=total,
                eval_steps=eval_steps, count_mode=count_mode, **kwargs
            )
            dev_metrics = self.eval_func(model, epoch=epoch+1)
        if dev_data is not None and self.evaluate_after:
            if self.checkpoint_path is not None and not self.save_all_checkpoints:
                model.load_state_dict(torch.load(self.checkpoint_path))
            self.do_epoch(model, dev_data, mode="validate", epoch="evaluate",
                          total=dev_total, count_mode=count_mode, **kwargs)
        return

    def is_better_score(self, epoch_score, best_score):
        if epoch_score is None:
            return False
        return (self.less_is_better == (epoch_score <= best_score))
    
    def evaluate_and_save_model(self, model, dev_data, epoch=None, total=None, **kwargs):
        if dev_data is not None:
            dev_metrics = self.do_epoch(model, dev_data, mode="validate", epoch=epoch, total=total, **kwargs)
            epoch_score = dev_metrics.get(self.validate_metric)
            to_save_checkpoint = self.save_all_checkpoints
            if self.is_better_score(epoch_score, self.best_score):
                to_save_checkpoint, self.best_score = True, epoch_score
        else:
            dev_metrics, to_save_checkpoint = None, True
        if to_save_checkpoint and self.checkpoint_path is not None:
            path_to_save = (attach_index(self.checkpoint_path, epoch, "\.pt") if self.save_all_checkpoints
                            else self.checkpoint_path)
            torch.save(model.state_dict(), path_to_save)
        return dev_metrics