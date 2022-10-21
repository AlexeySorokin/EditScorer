def binary_sequence_metrics(labels, pred_labels):
    answer = {
        "correct": 0, "total": len(labels), "TP": 0, "FP": 0, "FN": 0, "seq_total": 1,
        "seq_correct": int(labels == pred_labels)
    }
    # print(labels)
    # print(pred_labels.tolist())
    for x, y in zip(labels, pred_labels):
        answer["correct"] += int(x == y)
        answer["TP"] += int(x == y and x == 1)
        answer["FP"] += int(x != y and x == 0)
        answer["FN"] += int(x != y and x == 1)
    # print(answer)
    return answer

def aggregate_binary_sequence_metrics(metrics, alpha=1, recall_estimate=None):
    metrics["precision"] = metrics["TP"] / max(metrics["TP"] + metrics["FP"], 1)
    metrics["recall"] = metrics["TP"] / max(metrics["TP"] + metrics["FN"], 1)
    f_denominator = (metrics["FP"] + alpha ** 2 * metrics["FN"]) / (1 + alpha ** 2)
    metrics["F"] = metrics["TP"] / max(metrics["TP"] + f_denominator, 1)
    if recall_estimate is not None:
        FN = (metrics["FN"] + (1 - recall_estimate) * metrics["TP"]) / recall_estimate
        f_denominator = (metrics["FP"] + alpha ** 2 * FN) / (1 + alpha ** 2)
        metrics["recall_estimate"] = metrics["recall"] * recall_estimate
        metrics["F_estimate"] = metrics["TP"] / max(metrics["TP"] + f_denominator, 1)
    metrics["accuracy"] = metrics["correct"] / metrics["total"]
    metrics["seq_accuracy"] = metrics["seq_correct"] / metrics["seq_total"]
    return metrics

def display_metrics(metrics, metrics_to_display=None, percent_metrics=None, precision=4, only_main_loss=True):
    metrics_to_display = set(metrics_to_display or [])
    if only_main_loss:
        metrics_to_display.add("loss")
    else:
        metrics_to_display.update([metric for metric in metrics if "loss" in metric])
    percent_metrics = set(percent_metrics or [])
    for key in ["precision", "recall", "F", "accuracy"]:
        if key in metrics:
            metrics_to_display.add(key)
            percent_metrics.add(key)
    postfix = dict()
    for key in metrics_to_display:
        if key not in metrics:
            Warning(f"metric {key} is not present.")
        elif key in percent_metrics:
            postfix[key] = round(100 * metrics[key], max(precision-2, 0))
        else:
            postfix[key] = round(metrics[key], precision)
    return postfix