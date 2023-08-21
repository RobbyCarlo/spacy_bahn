from random import random

from pyarrow import json


def sample_jsonl(input_path, output_path, ratio=0.2, seed=15):
    if ratio < 0.0 or ratio > 1.0:
        print("Wrong format for portion. It should be between 0 and 1")
        return 0

    temp_list = []

    for line in open(input_path, "r"):
        l = json.loads(line)
        temp_list.append(l)
    n = int(ratio * len(temp_list))
    random.seed(seed)
    temp_list = random.sample(temp_list, k=n)
    with open(output_path, "w") as f:
        for element in temp_list:
            f.write(json.dumps(element) + "\n")
    return 0