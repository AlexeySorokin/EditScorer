from common.metrics import aggregate_binary_sequence_metrics


def extract_labels(model_output, batch):
    offsets = batch["offset"]
    y_pred = [model_output["probs"][start:end].tolist() for start, end in zip(offsets[:-1], offsets[1:])]
    y_true = []
    for index, start, end, default_index in zip(batch["indexes"], offsets[:-1], offsets[1:], batch["default"]):
        y_true.append({
            "label": batch["label"][start:end].cpu().tolist() if "label" in batch else None,
            "default": default_index, "index": index
        })
    # print(len(y_pred), len(y_true))
    return y_pred, y_true

def probs_to_labels(probs, default_index, threshold=0.5):
    default_prob = probs[default_index] if default_index is not None else 0.0
    pos_threshold = max(threshold, default_prob)
    return [int(prob >= (pos_threshold if i != default_index else threshold)) for i, prob in enumerate(probs)]

def item_score_func(y_true, y_pred, threshold=0.5, from_labels=False):
    TP, FN, FP, correct = 0, 0, 0, 0
    if not from_labels:
        y_pred = probs_to_labels(y_pred, y_true["default"], threshold)
    for index, (label, pred_label) in enumerate(zip(y_true["label"], y_pred)):
        is_correct = int(label == pred_label)
        if index != y_true["default"]:
            if label == 1:
                TP, FN = TP+is_correct, FN+(1-is_correct)
            elif pred_label == 1:
                FP += 1
        correct += is_correct
    all_correct = int(correct == len(y_pred))
    return {"TP": TP, "FP": FP, "FN": FN, "total": len(y_pred), "correct": correct,
            "seq_correct": all_correct, "seq_total": 1}


def evaluate_predictions(predictions, dataset, threshold=0.5, from_labels=True):
    metrics = dict()
    label_key = "labels" if from_labels else "probs"
    for curr_answer, curr_data in zip(predictions, dataset):
        y_true = {"label": [x["label"] for x in curr_data["data"]], "default": curr_data["default"]}
        curr_metrics = item_score_func(y_true, curr_answer[label_key], from_labels=from_labels, threshold=threshold)
        for key, value in curr_metrics.items():
            metrics[key] = metrics.get(key, 0) + value
        aggregate_binary_sequence_metrics(metrics, alpha=0.5)
    return metrics
    