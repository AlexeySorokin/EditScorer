import json
from argparse import ArgumentParser

import os
from itertools import chain

import numpy as np
import torch
import time

from errant_wrapper.read_errant import read_m2_simple
from ranking.data import prepare_dataset
from ranking.metrics import evaluate_predictions
from ranking.model import VariantScoringModel, VariantScoringModelWithAdditionalLayers, predict_with_model, \
    VariantScoringModelWithCrossAttention
from ranking.score_utils import extract_edits, score_corrections, print_corrections, make_output_data
from utils.data_utils import read_test_file
from utils.utils import make_offsets, apply_simple_edit

argument_parser = ArgumentParser()
argument_parser.add_argument("-m", "--model", default="roberta-base")
argument_parser.add_argument("-i", "--input_file", default="data/english/dev.bea.m2")
argument_parser.add_argument("--language", default=None)
argument_parser.add_argument("-r", "--raw", action="store_true")
argument_parser.add_argument("-E", "--evaluate", action="store_false")
argument_parser.add_argument("-v", "--input_variant_file", default="data/english_reranking/dev_1.bea.variants")
argument_parser.add_argument("-n", "--max_sents", default=None, type=int)
argument_parser.add_argument("-c", "--checkpoint_dir", type=str, required=True)
argument_parser.add_argument("-C", "--checkpoint_name", type=str, default="checkpoint.pt")
argument_parser.add_argument("-o", "--output_file", default=None, type=str)
argument_parser.add_argument("-O", "--output_dir", default="dump/reranking", type=str)
argument_parser.add_argument("-b", "--batch_size", default=4000, type=int)
argument_parser.add_argument("--threshold", default=0.5, type=float)
argument_parser.add_argument("-D", dest="use_default", action="store_false")
argument_parser.add_argument("-U", dest="use_position", action="store_false")
argument_parser.add_argument("-a", "--alpha_source", default=None, type=float, nargs="+")
argument_parser.add_argument("--max_source_score", default=5.0, type=float)
argument_parser.add_argument("--annotator_index", default=0, type=int)
argument_parser.add_argument("--min_diff", default=None, type=float)
argument_parser.add_argument("--seed", default=117, type=int)
argument_parser.add_argument("--thresholds", default=None, type=float, nargs="+")
argument_parser.add_argument("--min_dist", default=None, type=int, nargs="+")
argument_parser.add_argument("--n_max", default=None, type=int, nargs="+")

if __name__ == "__main__":
    args = argument_parser.parse_args()
    args.evaluate = args.evaluate and (not args.raw)
    if args.thresholds is None:
        args.thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if args.min_dist is None:
        args.min_dist = [1]
    if args.n_max is None:
        args.n_max = [4]
    config_path = os.path.join(args.checkpoint_dir, "config.json")
    with open(config_path, "r", encoding="utf8") as fin:
        config = json.load(fin)
    config.pop("epochs")
    if "cls" in config:
        cls = eval(config.pop("cls"))
    else:
        if config.get("n_attention_layers"):
            cls = VariantScoringModelWithAdditionalLayers
        elif config.get("cross_attention"):
            cls = VariantScoringModelWithCrossAttention
        else:
            cls = VariantScoringModel
    model = cls(model=args.model, device="cuda", use_position=args.use_position, **config)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_name)))
    torch.manual_seed(args.seed)
    # reading data
    if args.raw:
        source_data = [{"source": sent} for sent in read_test_file(args.input_file, n=args.max_sents)]
    else:
        source_data = read_m2_simple(args.input_file, n=args.max_sents)
        for elem in source_data:
            elem["edits"] = elem["edits"][args.annotator_index if (args.annotator_index < len(elem)) else 0]
            if not args.use_default:
                elem["edits"] = [edit for edit in elem["edits"] if edit.start >= 0]
            
    dataset_args = {
        "wrap_empty_edits": model.use_origin, "use_default": args.use_default,
        "use_sep_for_default": model.use_origin,
        "language": args.language, "min_diff": args.min_diff
    }
    dev_dataset, dev_data = prepare_dataset(
        args.input_variant_file, model=args.model, n=args.max_sents,
        return_flat=False, has_answers=(not args.raw), **dataset_args
    )
    sentence_offsets = [make_offsets([len(sent["words"]) for sent in elem]) for elem in dev_data]
    t1 = time.time()
    predictions = predict_with_model(model, dev_dataset, batch_size=args.batch_size, threshold=args.threshold)
    t2 = time.time()
    print(f"Elapsed time {t2-t1:.2f}")
    outdir = f"{args.output_dir}/{os.path.split(args.checkpoint_dir)[-1]}"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "edits_log.out"), "w", encoding="utf8") as fout:
        for curr_elem, curr_predictions in zip(chain.from_iterable(dev_data), predictions):
            assert len(curr_elem["edits"]) == len(curr_predictions["probs"])
            print(" ".join(curr_elem["words"]), file=fout)
            for edit, score in zip(curr_elem["edits"], curr_predictions["probs"]):
                print(edit["start"], edit["end"], end=" ", file=fout)
                if edit["start"] >= 0:
                    print(f"{edit['source'] or '_'} -> {edit['target'] or '_'}\t{edit['diff']:.2f}",
                          end=" ", file=fout)
                print(f"{score: .2f}", end=" ", file=fout)
                if "is_correct" in edit:
                    print(edit['is_correct'], end=" ", file=fout)
                print("", file=fout)
            print("", file=fout)
    if args.alpha_source is None:
        args.alpha_source = []
    args.alpha_source = [0.0] + sorted(set(args.alpha_source))
    for alpha in args.alpha_source:
        for curr_elem, curr_predictions in zip(chain.from_iterable(dev_data), predictions):
            assert len(curr_elem["edits"]) == len(curr_predictions["probs"])
            for edit, prob in zip(curr_elem["edits"], curr_predictions["probs"]):
                edit["score"] = np.log(prob) + alpha * min(edit.get("diff", 0.0), args.max_source_score)
        if alpha != 0.0:
            print(f"Source model weights={alpha:.2f}")
        # scoring data
        for threshold in args.thresholds:
        # for threshold in [0.7, 0.8, 0.9]:
            alpha_threshold = np.log(threshold) +  alpha
            to_dump = []
            if args.evaluate:
                metrics = evaluate_predictions(predictions, dev_dataset, from_labels=False, threshold=threshold)
                print(threshold, metrics)
            for min_dist in args.min_dist:
                for n_max in args.n_max:
                    outfile = os.path.join(outdir, f"{threshold:.1f}_{n_max}_{min_dist}.out")
                    if alpha != 0.0:
                        outfile = outfile[:-4] + f"_alpha={alpha:.2f}" + ".out"
                    # edits
                    extracted = [extract_edits(elem, only_generated=True, key="score", threshold=alpha_threshold,
                                               n_max=n_max, min_dist=min_dist)
                                 for elem in dev_data]
                    # printing output
                    print_corrections(source_data, dev_data, extracted, outfile)
                    # metrics
                    if args.evaluate:
                        print(f"min_dist={min_dist}")
                        TP = sum(int(edit["is_correct"]) for elem in chain.from_iterable(extracted) for edit in elem)
                        pred = sum(len(elem) for elem in chain.from_iterable(extracted))
                        total = sum(int(edit.start >= 0) for elem in source_data for edit in elem["edits"])
                        to_save = {
                            "alpha": alpha, "threshold": threshold, "alpha_threshold": alpha_threshold,
                            "max_edits": n_max,  "TP": TP, "FP": pred-TP, "FN": total-TP,
                            "P": round(100 * TP / max(pred, 1), 2), "R": round(100 * TP / max(total, 1), 2),
                            "F": round(100 * TP / max(0.2 * total + 0.8 * pred, 1), 2)
                        }
                        for key, value in to_save.items():
                            print(f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}", end=" ")
                        print("")
                        to_dump.append(to_save)
                    # generating output data
                    output_data = make_output_data(source_data, extracted, sentence_offsets)
                    outfile = os.path.join(outdir, f"{threshold:.1f}_{n_max}_{min_dist}.output")
                    if alpha != 0.0:
                        outfile = outfile[:-7] + f"_alpha={alpha:.2f}" + ".output"
                    with open(outfile, "w", encoding="utf8") as fout:
                        for sent in output_data:
                            print(*sent, file=fout)
                if args.evaluate:
                    print("")
            
    with open(os.path.join(outdir, "scoring.jsonl"), "w", encoding="utf8") as fout:
        for elem in to_dump:
            print(json.dumps(elem) + "\n", file=fout)
            # print(f"threshold={threshold}, max_edits={n_max}, TP={TP}, FP={FP}, FN={FN}",
            #       f"P={((100 * TP / (TP + FP))):.2f}, R={(100 * TP / (TP + FN)):.2f}, F1={(100 * TP / (TP + 0.2 * FN + 0.8 * FP)):.2f}")
        
        
        