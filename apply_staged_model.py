import json
from argparse import ArgumentParser

import os
from itertools import chain

import jsonlines
import numpy as np
import torch
import time

from errant_wrapper.read_errant import read_m2_simple
from ranking.data import prepare_dataset
from ranking.metrics import evaluate_predictions
from ranking.model import VariantScoringModel, VariantScoringModelWithAdditionalLayers, predict_with_model, \
    VariantScoringModelWithCrossAttention
from ranking.score_utils import extract_edits, score_corrections, predict_by_stages, print_corrections, make_output_data
from utils.data_utils import read_test_file
from utils.utils import make_offsets, apply_simple_edit

argument_parser = ArgumentParser()
argument_parser.add_argument("-m", "--model", default="roberta-base")
argument_parser.add_argument("-i", "--input_file", default="data/english/dev.bea.m2")
argument_parser.add_argument("-r", "--raw", action="store_true")
argument_parser.add_argument("-s", "--stages", default=8, type=int)
argument_parser.add_argument("-v", "--input_variant_file", default="data/english_reranking/dev_1.bea.variants")
argument_parser.add_argument("-n", "--max_sents", default=None, type=int)
argument_parser.add_argument("-l", "--language", default=None)
argument_parser.add_argument("-c", "--checkpoint_dir", type=str, required=True)
argument_parser.add_argument("-C", "--checkpoint_name", type=str, default="checkpoint.pt")
argument_parser.add_argument("-o", "--output_file", default=None, type=str)
argument_parser.add_argument("-O", "--output_dir", default="dump/reranking", type=str)
argument_parser.add_argument("--output_by_stages", action="store_true")
argument_parser.add_argument("-b", "--batch_size", default=4000, type=int)
argument_parser.add_argument("--threshold", default=0.5, type=float)
argument_parser.add_argument("-T", "--stage_thresholds", default=None, type=float, nargs="+")
argument_parser.add_argument("-D", dest="use_default", action="store_false")
argument_parser.add_argument("-U", dest="use_position", action="store_false")
argument_parser.add_argument("-a", "--alpha_source", default=None, type=float, nargs="+")
argument_parser.add_argument("-A", "--threshold_alpha", default=1.0, type=float)
argument_parser.add_argument("--max_source_score", default=5.0, type=float)
argument_parser.add_argument("--annotator_index", default=0, type=int)
argument_parser.add_argument("--seed", default=117, type=int)


if __name__ == "__main__":
    args = argument_parser.parse_args()
    if args.stage_thresholds is None:
        args.stage_thresholds = [0.7, 0.8, 0.9]
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
    with jsonlines.open(args.input_variant_file) as fin:
        data = list(fin)
        if args.max_sents is not None:
            data = data[:args.max_sents]
        for elem in chain.from_iterable(data):
            if isinstance(elem["words"], str):
                elem["words"] = elem["words"].split()
            if not args.raw:
                elem["edits"] = [edit for edit in elem["edits"] if edit["diff"] is not None and edit["is_generated"]]
            for edit in elem["edits"]:
                if "words" not in edit:
                    edit["words"] = apply_simple_edit(elem["words"], edit["start"], edit["end"], edit["target"])
    flat_data = list(chain.from_iterable(data))
    offsets = make_offsets([len(x) for x in data])
    sentence_offsets = [make_offsets([len(x["words"]) for x in elem]) for elem in data]
    outdir = f"{args.output_dir}/{os.path.split(args.checkpoint_dir)[-1]}"
    os.makedirs(outdir, exist_ok=True)
    if args.alpha_source is None:
        args.alpha_source = []
    args.alpha_source = [0.0] + sorted(set(args.alpha_source))
    for alpha in args.alpha_source:
        print(f"Source model weights={alpha:.2f}")
        for threshold in args.stage_thresholds:
            outfile = os.path.join(outdir, f"{threshold:.1f}_staged.out")
            if alpha != 0.0:
                outfile = outfile[:-11] + f"_alpha={alpha:.2f}_{args.threshold_alpha:.2f}" + "_staged.out"
            alpha_threshold = np.log(threshold) + alpha * args.threshold_alpha
            t1 = time.time()
            extracted = predict_by_stages(model, flat_data, tokenizer=args.model, language=args.language,
                                          rounds=args.stages, n_max=1, threshold=alpha_threshold,
                                          alpha_source=alpha)
            t2 = time.time()
            print(f"Elapsed time {t2-t1:.6f}")
            for stage in range(1, args.stages+1):
                if stage == args.stages:
                    grouped_extracted = [extracted[i:j] for i, j in zip(offsets[:-1], offsets[1:])]
                    # printing output
                    print_corrections(source_data, data, grouped_extracted, outfile)
                elif args.output_by_stages:
                    stage_outfile = outfile[:-4] + f"_stage_{stage}.output"
                    stage_extracted = [[edit for edit in elem if edit["stage"] <= stage] for elem in extracted]
                    grouped_extracted = [stage_extracted[i:j] for i, j in zip(offsets[:-1], offsets[1:])]
                    output_data = make_output_data(source_data, grouped_extracted, sentence_offsets)
                    with open(stage_outfile, "w", encoding="utf8") as fout:
                        for sent in output_data:
                            print(*sent, file=fout)
                if not args.raw:
                    print(f"Stage {stage}")
                    TP = sum(int(edit["is_correct"] and edit["stage"] <= stage) for elem in extracted for edit in elem)
                    pred = sum(int(edit["stage"] <= stage) for elem in extracted for edit in elem)
                    total = sum(int(edit.start >= 0) for elem in source_data for edit in elem["edits"])
                    to_save = {
                        "alpha": alpha, "threshold": threshold, "alpha_threshold": alpha_threshold,
                        "TP": TP, "FP": pred - TP, "FN": total - TP,
                        "P": round(100 * TP / max(pred, 1), 2), "R": round(100 * TP / max(total, 1), 2),
                        "F": round(100 * TP / max(0.2 * total + 0.8 * pred, 1), 2)
                    }
                    for key, value in to_save.items():
                        print(f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}", end=" ")
                    print("")
            # to_dump.append(to_save)
            # generating output data
            output_data = make_output_data(source_data, grouped_extracted, sentence_offsets)
            outfile = os.path.join(outdir, f"{threshold:.1f}_staged.output")
            if alpha != 0.0:
                outfile = outfile[:-14] + f"_alpha={alpha:.2f}_{args.threshold_alpha:.2f}" + "_staged.output"
            with open(outfile, "w", encoding="utf8") as fout:
                for sent in output_data:
                    print(*sent, file=fout)
        print("")
        
        