from itertools import chain

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import ModuleList
from torch.nn.functional import binary_cross_entropy
from tqdm.auto import tqdm
from transformers import AutoModel, AdamW, get_scheduler
from transformers.models.roberta.modeling_roberta import RobertaLayer

from ranking.data import prepare_dataloader
from ranking.metrics import probs_to_labels, extract_labels


def pairwise_loss(logits, pairs, margin, reduce="mean", epsilon=1e-6, from_log_probs=False):
    if from_log_probs:
        neg_logits = torch.log(torch.clamp(1 - torch.exp(logits), epsilon, 1.0))
        logits = logits - neg_logits
    if len(pairs) > 0:
        answer = torch.relu(margin - logits[pairs[:,0]] + logits[pairs[:,1]])
    else:
        answer = torch.zeros_like(logits)
    if reduce == "mean":
        answer = answer.mean()
    elif reduce == "sum":
        answer = answer.sum()
    return answer

def threshold_contrastive_loss(logits, labels, default_index):
    logits = logits - logits.max()  # to avoid large exponents
    exponents = torch.exp(logits)
    if not(labels.any()) or labels.all():  # no negative labels
        pos_loss = torch.zeros(size=(0,), dtype=logits.dtype, device=logits.device)
    else:
        # \forall i \in Pos -(\pi_i - \log(\sum_{j \in Neg} \pi_j + \exp(\pi_i)))
        pos_loss = torch.log(torch.sum(exponents[~labels]) + exponents[labels]) - logits[labels]
    neg_labels = ~labels
    neg_labels[default_index] = True
    if neg_labels.any():
        neg_loss = torch.log(exponents[neg_labels] + exponents[default_index]) - logits[default_index]
    else:
        neg_loss = torch.zeros(size=(0,), dtype=logits.dtype, device=logits.device)
    return pos_loss, neg_loss

def bce_group_loss(probs, labels, alpha_pos=1.0, loss_by_class=True):
    if loss_by_class:
        loss = binary_cross_entropy(probs, labels.float(), reduction="none")
        pos_loss = loss[labels.bool()].mean() if labels.any() else torch.as_tensor(0.0)
        neg_loss = loss[~(labels.bool())].mean() if (1 - labels).any() else torch.as_tensor(0.0)
        bce_loss = (alpha_pos * pos_loss + neg_loss) / (alpha_pos + 1)
    else:
        weights = torch.where(labels.bool(), alpha_pos, 1.0)
        bce_loss = binary_cross_entropy(probs, labels.float(), weight=weights, reduction="sum") / weights.sum()
    return bce_loss


class VariantScoringModel(nn.Module):

    def __init__(self, model, model_field="last_hidden_state", mlp_hidden=None, mlp_dropout=0.0,
                 use_position=True, position_mode="first", use_origin=False, concat_mode=None,
                 loss_by_class=False, average_loss_for_batch=True,
                 alpha_pos=1.0, alpha_soft=0.0, alpha_hard=0.0,
                 alpha_no_change=0.0, soft_margin=2.0, hard_margin=2.0, no_change_margin=2.0,
                 alpha_contrastive=0.0, contrastive_loss_by_classes=True,
                 lr=1e-5, clip=None, scheduler="constant", warmup=0, batches_per_update=1,
                 device="cuda", **kwargs):
        super(VariantScoringModel, self).__init__()
        if isinstance(model, str):
            model = AutoModel.from_pretrained(model).to(device)
        self.model = model
        self.model_field = model_field
        self.mlp_hidden = mlp_hidden or []
        self.mlp_dropout = mlp_dropout
        self.use_position = use_position
        self.position_mode = position_mode
        self.use_origin = use_origin
        self.concat_mode = concat_mode
        self.build_network(**kwargs)
        self.loss_by_class = loss_by_class
        self.average_loss_for_batch = average_loss_for_batch
        self.alpha_pos = alpha_pos
        self.alpha_soft = alpha_soft
        self.alpha_hard = alpha_hard
        self.alpha_no_change = alpha_no_change
        self.soft_margin = soft_margin
        self.hard_margin = hard_margin
        self.no_change_margin = no_change_margin
        self.alpha_contrastive = alpha_contrastive
        self.contrastive_loss_by_classes = contrastive_loss_by_classes
        self.lr = lr
        self.clip = clip
        self.device = device
        self.batches_per_update = batches_per_update
        self._batches_accumulated = 0
        if self.device is not None:
            self.to(self.device)
        optimizer_args = {key[10:]: value for key, value in kwargs.items() if key[:10] == "optimizer_"}
        self.build_optimizer(**optimizer_args)
        self.scheduler = get_scheduler(scheduler, optimizer=self.optimizer, num_warmup_steps=warmup)
    
    @property
    def max_length(self):
        return self.model.config.max_position_embeddings

    @property
    def model_hidden_size(self):
        mult = 1 if not self.use_origin else 4 if self.concat_mode == "infersent" else 2
        return mult * self.model.config.hidden_size

    @property
    def state_size(self):
        return self.mlp_hidden[-1] if len(self.mlp_hidden) > 0 else self.model_hidden_size
    
    def build_optimizer(self, **kwargs):
        self.optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
    
    def build_network(self, **kwargs):
        self.mlp = ModuleList()
        prev_hidden = self.model_hidden_size
        for hidden in self.mlp_hidden:
            self.mlp.append(torch.nn.Linear(prev_hidden, hidden))
            self.mlp.append(torch.nn.ReLU())
            if self.mlp_dropout > 0.0:
                self.mlp.append(torch.nn.Dropout(self.mlp_dropout))
        self.proj_layer = torch.nn.Linear(self.state_size, 1)
        return
    
    def encoder(self, input_ids, **kwargs) -> Tensor:
        answer = self.model(input_ids)
        return answer[self.model_field]
    
    def _reduce_state(self, state):
        if self.position_mode == "mean":
            return state.mean(dim=-2)
        elif self.position_mode == "last":
            return state[-1]
        return state[0]
    
    def forward(self, input_ids, start, end, origin_start, origin_end, return_sigmoid=True, **kwargs):
        encodings = self.encoder(input_ids, **kwargs)
        if self.use_position:
            states = [elem[curr_start:curr_end] for elem, curr_start, curr_end in zip(encodings, start, end)]
            states = torch.stack([self._reduce_state(state) for state in states], dim=0)
            if self.use_origin:
                origin_states = [elem[curr_start:curr_end]
                                 for elem, curr_start, curr_end in zip(encodings, origin_start, origin_end)]
                origin_states = torch.stack([self._reduce_state(state) for state in origin_states], dim=0)
                to_concat = [states, origin_states]
                if self.concat_mode == "infersent":
                    to_concat.extend([states-origin_states, states*origin_states])
                states = torch.cat(to_concat, dim=-1)
        else:
            states = encodings[:,0]
        for layer in self.mlp:
            states = layer(states)
        logits = self.proj_layer(states)[:,0]
        return torch.sigmoid(logits) if return_sigmoid else logits

    def train_on_batch(self, batch, mask=None):
        self.train()
        if self._batches_accumulated == 0:
            self.optimizer.zero_grad()
        loss = self._validate(**batch, mask=mask)
        loss["loss"] /= self.batches_per_update
        loss["loss"].backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
        self._batches_accumulated = (self._batches_accumulated + 1) % self.batches_per_update
        if self._batches_accumulated == 0:
            self.optimizer.step()
            self.scheduler.step()
        return loss

    def validate_on_batch(self, batch, mask=None):
        self.eval()
        with torch.no_grad():
            return self._validate(**batch, mask=mask)

    def _validate(self, input_ids, label, soft_pairs, hard_pairs, no_change_pairs=None, mask=None, **kwargs):
        if self.device is not None:
            label = label.to(self.device)
        logits = self(input_ids, return_sigmoid=False, **kwargs) #   self.forward(x) = self.__call__(x)
        probs = torch.sigmoid(logits)
        offsets, default_indexes = kwargs["offset"], kwargs["default"]
        if self.average_loss_for_batch:
            bce_loss = bce_group_loss(probs, label, alpha_pos=self.alpha_pos, loss_by_class=self.loss_by_class)
        else:
            bce_losses = [
                bce_group_loss(probs[start:end], label[start:end],
                         alpha_pos=self.alpha_pos, loss_by_class=self.loss_by_class)
                for start, end in zip(offsets[:-1], offsets[1:])
            ]
            bce_loss = torch.stack(bce_losses, dim=0).mean()
        answer = {"bce_loss": bce_loss, "probs": probs}
        if self.alpha_contrastive:
            pos_losses, neg_losses = [], []
            for start, end, default in zip(offsets[:-1], offsets[1:], default_indexes):
                curr_pos_losses, curr_neg_losses =\
                    threshold_contrastive_loss(logits[start:end], label[start:end].bool(), default)
                pos_losses.append(curr_pos_losses)
                neg_losses.append(curr_neg_losses)
            pos_losses, neg_losses = torch.cat(pos_losses, dim=0), torch.cat(neg_losses, dim=0)
            if self.contrastive_loss_by_classes:
                pos_loss = pos_losses.mean() if pos_losses.shape[0] > 0 else torch.as_tensor(0.0)
                neg_loss = neg_losses.mean() if neg_losses.shape[0] > 0 else torch.as_tensor(0.0)
                contrastive_loss = (pos_loss + neg_loss) / 2.0
            else:
                contrastive_loss = torch.cat([pos_losses, neg_losses], dim=0).mean()
            loss = bce_loss + self.alpha_contrastive * contrastive_loss
            answer.update({"loss": loss, "c_loss": contrastive_loss})
        else:
            soft_loss = pairwise_loss(logits, soft_pairs, self.soft_margin)
            hard_loss = pairwise_loss(logits, hard_pairs, self.hard_margin)
            no_change_loss = pairwise_loss(logits, no_change_pairs, self.no_change_margin)
            loss = bce_loss + self.alpha_soft * soft_loss + self.alpha_hard * hard_loss +\
                   self.alpha_no_change * no_change_loss
            answer.update({
                "loss": loss, "soft_loss": soft_loss, "hard_loss": hard_loss, "no_change_loss": no_change_loss
            })
        return answer
    
    
def predict_with_model(model, dataset, batch_size=1500, device="cuda", threshold=0.5):
    answer = [None] * len(dataset)
    short_dataset, long_indexes, zero_indexes = [], [], []
    for i, elem in enumerate(dataset):
        if len(elem["data"]) == 0:
            zero_indexes.append(i)
        elif max(len(x["input_ids"]) for x in elem["data"]) <= model.max_length:
            short_dataset.append(elem)
        else:
            long_indexes.append(i)
    dataloader = prepare_dataloader(short_dataset, batch_size=batch_size, device=device)
    progress_bar = tqdm(total=len(short_dataset), leave=True)
    with progress_bar:
        for batch in dataloader:
            model.eval()
            with torch.no_grad():
                batch_output = {"probs": model(**batch)}
            probs, batch_info = extract_labels(batch_output, batch)
            for curr_probs, elem in zip(probs, batch_info):
                answer[elem["index"]] = {
                    "probs": curr_probs, "labels": probs_to_labels(curr_probs, elem["default"], threshold=threshold)
                }
            progress_bar.update(len(probs))
    for i in long_indexes:
        elem = dataset[i]
        diffs = np.array([x.get("diff", 0.0) for x in elem["data"]])
        probs = 1 / (1.0 + np.exp(-diffs))
        answer[i] = {"probs": probs, "labels": probs_to_labels(probs, elem["default"], threshold=threshold)}
    for i in zero_indexes:
        answer[i] = {"probs": [], "labels": []}
    return answer


class VariantScoringModelWithAdditionalLayers(VariantScoringModel):
    
    def build_optimizer(self, attention_lr=None, **kwargs):
        if attention_lr is not None:
            self.optimizer = AdamW([
                {"params": self.attention_layers.parameters(), "lr": attention_lr},
                {"params": chain(self.model.parameters(), self.mlp.parameters(), self.proj_layer.parameters())}
            ], lr=self.lr, weight_decay=0.01)
        else:
            self.optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
    
    def build_network(self, n_attention_layers=0, residual=False, init_with_last_layer=False, **kwargs):
        self.n_attention_layers = n_attention_layers
        self.residual = residual
        self.attention_layers = ModuleList()
        for i in range(self.n_attention_layers):
            layer = RobertaLayer(self.model.config)
            if init_with_last_layer:
                layer_params = dict(layer.named_parameters())
                for name, param in self.model.encoder.layer[i-self.n_attention_layers].named_parameters():
                    layer_params[name].data.copy_(param.data)
            self.attention_layers.append(layer)
        super(VariantScoringModelWithAdditionalLayers, self).build_network(**kwargs)
        return
    
    def encoder(self, input_ids, **kwargs) -> Tensor:
        encodings = super(VariantScoringModelWithAdditionalLayers, self).encoder(input_ids)
        if self.n_attention_layers == 0:
            return encodings
        for layer in self.attention_layers:
            new_encodings = layer(encodings)[0] # takes 0-th element since `RobertaLayer` outputs a tuple
            encodings = (encodings + new_encodings) if self.residual else new_encodings
        return encodings
    
    
class VariantScoringModelWithCrossAttention(VariantScoringModel):
    
    def build_network(self, residual=False, **kwargs):
        super(VariantScoringModelWithCrossAttention, self).build_network(**kwargs)
        self.residual = residual
        self.cross_attention_layer = RobertaLayer(self.model.config)
        return

    def forward(self, input_ids, start, end, offset, return_sigmoid=True, **kwargs):
        encodings = self.encoder(input_ids, **kwargs)
        if self.use_position:
            states = [elem[curr_start:curr_end] for elem, curr_start, curr_end in zip(encodings, start, end)]
            states = torch.stack([self._reduce_state(state) for state in states], dim=0)
        else:
            states = encodings[:,0]
        # states.shape = (B, 768)
        new_states = []
        for i, j in zip(offset[:-1], offset[1:]):
            new_states.append(self.cross_attention_layer(states[i:j].unsqueeze(dim=0))[0][0])
        new_states = torch.cat(new_states, dim=0)
        if self.residual:
            states = states + new_states
        else:
            states = new_states
        for layer in self.mlp:
            states = layer(states)
        logits = self.proj_layer(states)[:,0]
        return torch.sigmoid(logits) if return_sigmoid else logits
