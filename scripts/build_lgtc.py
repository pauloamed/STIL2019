# Converts corpus on conll format to the macmorpho format
import sys
import itertools
from itertools import combinations, chain

data_path = "../data/"

FILES_UD = [
    'pt_bosque-ud-train.mm',
    'pt_bosque-ud-dev.mm',
    'pt_bosque-ud-test.mm',
]

FILE_LGTC = 'Bosque_CF_lgtc.mm'

FILES_DEST_LGTC = [
    'lgtc-train.mm',
    'lgtc-dev.mm',
    'lgtc-test.mm',
]

def open_file(file_name):
    print(">> Trying to open file...")
    try:
        f = open(file_name, "rt")
    except:
        print(">>>> Unable to open file")
        exit()
    print(">>> File was successfully opened")
    return f.read()


def get_sent_text(sent):
    return [t.split('_')[0] for t in sent.split(' ')]

def get_sents(text, tagged=False):
    return [sent if tagged else get_sent_text(sent) for sent in text.split('\n')]

def process_ud_file(data_path, file_name):
    text = open_file(data_path + file_name)
    sents = get_sents(text)

    sents_set = set()

    for s in sents:
        s = " ".join(s)
        if len(s) == 0:
            continue

        sents_set.add(s.strip())

    return sents_set

def set_sents(gone, lgtc_text, subset):
    ret = []

    for sent in lgtc_text:

        if sent in gone:
            continue

        sent_text = (" ".join(get_sent_text(sent))).strip()

        if sent_text in subset:
            gone.add(sent)
            ret.append(sent)
    return ret

def complete_set(gone, lgtc_text, lgtc_subset, subset_size):
    i = 0
    while len(lgtc_subset) < subset_size:
        # print(i, len(lgtc_text), len(lgtc_subset), subset_size)
        if lgtc_text[i] not in gone:
            lgtc_subset.append(lgtc_text[i])
            gone.add(lgtc_text[i])
        i += 1

def create_file(file_name, samples):
    try:
        file = open(file_name, "w")
    except:
        print(">>>> Unable to create file")
        exit()

    for sample in samples:
        if sample == []:
            continue
        sample = sample.split(' ')
        for i in range(len(sample)):
            file.write(sample[i])
            if i < len(sample)-1:
                file.write(" ")
        file.write("\n")
    file.close()

    print(">>> File was successfully created")

ud_train = process_ud_file(data_path, FILES_UD[0])
ud_dev = process_ud_file(data_path, FILES_UD[1])
ud_test = process_ud_file(data_path, FILES_UD[2])

lgtc_text =  get_sents(open_file(data_path + FILE_LGTC), tagged=True)
lgtc_text = list(set(lgtc_text))

train_size = int(len(lgtc_text) * 0.8)
dev_size = int(len(lgtc_text) * 0.1)
test_size = len(lgtc_text) - (train_size + dev_size)

gone = set()


lgtc_train = set_sents(gone, lgtc_text, ud_train)
lgtc_dev = set_sents(gone, lgtc_text, ud_dev)
lgtc_test = set_sents(gone, lgtc_text, ud_test)

complete_set(gone, lgtc_text, lgtc_train, train_size)
complete_set(gone, lgtc_text, lgtc_dev, dev_size)
complete_set(gone, lgtc_text, lgtc_test, test_size)

create_file(data_path + FILES_DEST_LGTC[0], lgtc_train)
create_file(data_path + FILES_DEST_LGTC[1], lgtc_dev)
create_file(data_path + FILES_DEST_LGTC[2], lgtc_test)
