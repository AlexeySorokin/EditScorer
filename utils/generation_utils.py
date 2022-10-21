import numpy as np

from utils.utils import word_to_error_span, apply_simple_edit, do_spans_overlap


def _finalize_correction(correction):
    correction = correction.copy()
    correction["start"], correction["end"] = correction["actual_start"], correction["actual_end"]
    correction["work_source"], correction["work_words"] = correction["source"], correction["words"]
    correction.pop("actual_start")
    correction.pop("actual_end")
    return correction

def extract_corrections(corrections, L, threshold=2.0, n_max=None, min_dist=5, add_default=False):
    answer = []
    are_covered = np.zeros(shape=(L,), dtype=bool)
    are_strictly_covered = np.zeros(shape=(2*L+1,), dtype=bool)
    for edit in corrections:
        if edit["start"] < 0:
            if add_default:
                answer.append(edit)
            are_covered[:] = True
            break
        if edit["diff"] < threshold:
            break
        span_start, span_end = word_to_error_span(edit["start"], edit["end"])
        if are_covered[edit["start"]:edit["end"]].any():
            continue
        if n_max is None or len(answer) < n_max:
            answer.append(edit)
        cover_start, cover_end = max(edit["start"]-min_dist, 0), min(edit["end"]+min_dist, L)
        are_covered[cover_start:cover_end] = True
        are_strictly_covered[span_start:span_end] = True
    return answer


def update_words_and_variants(correction, variants, words, target_key="correction"):
    new_variants = []
    offset = len(correction[target_key].split()) - correction["end"] + correction["start"]
    new_words = apply_simple_edit(words, correction["start"], correction["end"], correction[target_key])
    if len(new_words) > 0:
        for variant in variants:
            if do_spans_overlap((correction["start"], correction["end"]), (variant["start"], variant["end"])):
                continue
            variant = variant.copy()
            if variant["start"] >= correction["end"]:
                variant["start"] += offset
                variant["end"] += offset
            variant["source"] = new_words
            if variant[target_key] is not None:
                variant["words"] = apply_simple_edit(new_words, variant["start"], variant["end"], variant[target_key])
            else:
                variant["words"] = variant["source"]
            new_variants.append(variant)
    return new_variants, new_words