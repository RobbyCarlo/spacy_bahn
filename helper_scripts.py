from random import random
from sklearn.model_selection import train_test_split
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


def split_jsonl(input_path, output_train, output_test, output_val, ratio_test=0.3, ratio_val=0.5):

    temp_list = []
    for line in open(input_path, "r"):
        #l = json.loads(line)
        temp_list.append(line)

    train, test = train_test_split(temp_list, test_size=0.3)
    val, test = train_test_split(test, test_size=0.5)

    with open(output_train, "w") as f1:
        for element in train:
            f1.write(element)
            #f1.write(json.dumps(element) + "\n")

    with open(output_test, "w") as f2:
        for element in test:
            f2.write(element)
            #f2.write(json.dumps(element) + "\n")

    with open(output_val, "w") as f3:
        for element in val:
            f3.write(element)
            #f3.write(json.dumps(element) + "\n")

if __name__ == "__main__":

    split_jsonl(input_path="KundenKommunikation_ner_annotiert.jsonl",
                output_train="KundenKommunikation_ner_annotiert_train.jsonl",
                output_test="KundenKommunikation_ner_annotiert_test.jsonl",
                output_val="KundenKommunikation_ner_annotiert_val.jsonl")



