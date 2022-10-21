from argparse import ArgumentParser
import json

from tqdm.auto import tqdm

from utils.align import align_sentences

def read_infile(infile):
    answer = []
    with open(infile, "r", encoding="utf8") as fin:
        state = 0
        for i, line in enumerate(fin):
            line = line.strip()
            if state == 0:
                if not line.startswith("| Type the input sentence and press return:"):
                    continue
                curr_data = None
                state = 1
                continue
            if state == 2:
                if line.startswith("| WARNING"):
                    continue
                if line[:2] == "S-":
                    state = 1
                    answer.append(curr_data.copy())
                    curr_data = None
            if state == 1:
                curr_data = {"words": line.split()[1:], "edits": []}
                state = 2
            elif state == 2:
                _, score, words = line.split(maxsplit=2)
                curr_data["edits"].append({"score": float(score), "words": [x for x in words.split() if x != "<s>"]})
                state = 3
            elif state == 3:
                curr_data["edits"][-1]["word_scores"] = list(map(float, line.split()[2:]))
                curr_data["edits"][-1]["total_score"] = sum(curr_data["edits"][-1]["word_scores"])
                state = 2
        if curr_data is not None:
            answer.append(curr_data)
    return answer


def find_longest_prefix(first, second):
    if len(first) > len(second):
        first, second = second, first
    for i, (a, b) in enumerate(zip(first, second)):
        if a != b:
            return i
    return len(first)

def find_difference_span(first, second):
    start = find_longest_prefix(first, second)
    suffix = find_longest_prefix(first[start:][::-1], second[start:][::-1])
    target = second[start:len(second) - suffix]
    return start, len(first) - suffix, target

def annotate_single(sent_data):
    default_score = None
    for elem in sent_data["edits"]:
        elem["total_score"] = sum(elem["word_scores"])
        if elem["words"] == sent_data["words"]:
            default_score = elem["total_score"]
            elem["start"], elem["end"] = -1, -1
            elem["source"] = ""
            elem["target"] = None
        else:
            elem["start"], elem["end"], target = find_difference_span(sent_data["words"], elem["words"])
            elem["source"] = " ".join(sent_data["words"][elem["start"]:elem["end"]])
            elem["target"] = " ".join(target)
    if default_score is None:
        default_score = min(elem["total_score"] for elem in sent_data["edits"]) - 1.0
    for elem in sent_data["edits"]:
        elem["diff"] = elem["total_score"] - default_score
    return sent_data

def annotate(sent_data):
    source_words = sent_data["words"]
    hypo_triples, all_triples = [], set()
    for elem in sorted(sent_data["edits"], key=lambda x: x["total_score"], reverse=True):
        target_words = elem["words"]
        curr_triples = align_sentences(source_words, target_words)
        hypo_triples.append(curr_triples)
        all_triples.update(curr_triples)
    first_uncovering_hypos = [0] * (2 * len(source_words) + 1)
    pos_scores = dict()
    for i, curr_triples in enumerate(hypo_triples):
        for triple in curr_triples:
            if triple not in pos_scores:
                pos_scores[triple] = sent_data["edits"][i]["total_score"]
            new_start = 2 * triple[0] + int(triple[1] > triple[0])
            new_end = 2 * triple[1] + int(triple[1] == triple[0])
            for pos in range(new_start, new_end):
                if first_uncovering_hypos[pos] == i:
                    first_uncovering_hypos[pos] += 1
    scores = [hypo["total_score"] for hypo in sent_data["edits"]] + [sent_data["edits"][-1]["total_score"] - 1.0]
    diffs = dict()
    for triple, pos_score in pos_scores.items():
        new_start = 2 * triple[0] + int(triple[1] > triple[0])
        new_end = 2 * triple[1] + int(triple[1] == triple[0])
        diffs[triple] = pos_score - scores[max(first_uncovering_hypos[new_start:new_end])]
    answer = []
    for triple, diff in sorted(diffs.items(), key=lambda x: x[1], reverse=True):
        start, end, candidate = triple
        curr_answer = {
            "start": start, "end": end,
            "source": " ".join(source_words[start:end]), "target": candidate,
            "diff": diff
        }
        answer.append(curr_answer)
    curr_answer = {
        "start": -1, "end": -1, "target": None, "source": "",
        "words": source_words, "diff": 0.0,
    }
    answer.append(curr_answer)
    return {"words": sent_data["words"], "edits": answer}

argument_parser = ArgumentParser()
argument_parser.add_argument("-i", "--infile", required=True)
argument_parser.add_argument("-o", "--outfile", required=True)
argument_parser.add_argument("-s", "--single", action="store_true")


if __name__ == "__main__":
    args = argument_parser.parse_args()
    data = read_infile(args.infile)
    func = annotate_single if args.single else annotate
    answer = [func(sent_data) for sent_data in tqdm(data)]
    with open(args.outfile, "w", encoding="utf8") as fout:
        for sent_data in answer:
            # print(*(sent_data["words"]))
            # for edit in sent_data["edits"]:
            #     source = " ".join(sent_data["words"][edit["start"]:edit["end"]]) or ""
            #     target = edit["target"] or ""
            #     print(edit["start"], edit["end"] , f"{source} -> {target} {edit['diff']:.2f}")
            # print("")
            print(json.dumps(sent_data), file=fout)