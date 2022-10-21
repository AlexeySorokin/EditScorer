from collections import defaultdict


from errant_wrapper.edit import Edit, apply_edits

def dump_sentence_annotation(curr_data):
    annotators_number = max(curr_data["edits"]) + 1
    curr_edits = [curr_data["edits"][i] for i in range(annotators_number)]
    # noinspection PyTypeChecker
    curr_corrections = [
        apply_edits(curr_data["source"], annotator_edits)["sent"] for annotator_edits in curr_edits
    ]
    answer = {
        "source": curr_data["source"], "correct": curr_corrections, "edits": curr_edits
    }
    return answer
    

def read_m2_simple(infile, n=None, lines_have_prefix=True):
    answer = []
    curr_data = {"edits": defaultdict(list, {0: []})}
    with open(infile, "r", encoding="utf8") as fin:
        mode = "S"
        for line in fin:
            if n is not None and len(answer) >= n:
                break
            line = line.strip()
            if line == "":
                if "source" in curr_data:
                    answer.append(dump_sentence_annotation(curr_data))
                    curr_data = {"edits": defaultdict(list, {0: []})}
                    mode = "S"
                continue
            if lines_have_prefix or mode != "S":
                mode, line = line[0], line[1:].strip()
            if mode == "S":
                if "source" in curr_data:
                    answer.append(dump_sentence_annotation(curr_data))
                    curr_data = {"edits": defaultdict(list, {0: []})}
                curr_data["source"] = line.split()
                mode = "A"
            else:
                splitted = line.split("|||")
                start, end = map(int, splitted[0].split())
                edit_type, correction, annotator = splitted[1], splitted[2], int(splitted[-1])
                curr_data["edits"][annotator].append(Edit(start, end, correction, edit_type, annotator))
    if "source" in curr_data:
        answer.append(dump_sentence_annotation(curr_data))
    return answer


def write_sentence_corrections(data, fout):
    print("S " + " ".join(data["source"]), file=fout)
    for annotator_edits in data["edits"]:
        for edit in annotator_edits:
            print("A " + str(edit), file=fout)
    print("", file=fout)
                    
                
    