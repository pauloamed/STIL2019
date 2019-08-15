# Converts corpus on conll format to the macmorpho format
import sys
import itertools
from itertools import combinations, chain

DATA_PATH = "../data/"

UD_TRAIN_FILE = 'pt_bosque-ud-train.mm.txt'
UD_DEV_FILE = 'pt_bosque-ud-dev.mm.txt'
UD_TEST_FILE = 'pt_bosque-ud-test.mm.txt'

FILE_LGTC = 'Bosque_CF_lgtc.mm.txt'

DEST_LGTC_TRAIN = 'lgtc-train.mm.txt'
DEST_LGTC_DEV = 'lgtc-dev.mm.txt'
DEST_LGTC_TEST = 'lgtc-test.mm.txt'


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

# opens files and loads to texts, then saves them on a hash for further usage
ud_train = process_ud_file(DATA_PATH, UD_TRAIN_FILE)
ud_dev = process_ud_file(DATA_PATH, UD_DEV_FILE])
ud_test = process_ud_file(DATA_PATH, UD_TEST_FILE)

# retrieves all tagged samples from one-file linguateca dataset
lgtc_text =  get_sents(open_file(DATA_PATH + FILE_LGTC), tagged=True)

# eliminates duplicates
lgtc_text = list(set(lgtc_text))

# calculating sizes of training, dev(Val) and test sets (following a 1/1/8) split
train_size = int(len(lgtc_text) * 0.8)
dev_size = int(len(lgtc_text) * 0.1)
test_size = len(lgtc_text) - (train_size + dev_size)

# hash for marking the already designated-to-sets samples
gone = set()

# uses the already built hashes for each set of Bosque-UD and designates the intersections
# to sets of linguateca's sets
lgtc_train = set_sents(gone, lgtc_text, ud_train)
lgtc_dev = set_sents(gone, lgtc_text, ud_dev)
lgtc_test = set_sents(gone, lgtc_text, ud_test)

# completes the linguateca's sets
complete_set(gone, lgtc_text, lgtc_train, train_size)
complete_set(gone, lgtc_text, lgtc_dev, dev_size)
complete_set(gone, lgtc_text, lgtc_test, test_size)

# creates linguatecas giles
create_file(DATA_PATH + DEST_LGTC_TRAIN, lgtc_train)
create_file(DATA_PATH + DEST_LGTC_DEV, lgtc_dev)
create_file(DATA_PATH + DEST_LGTC_TEST, lgtc_test)
