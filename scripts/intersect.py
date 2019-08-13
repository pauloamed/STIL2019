# Counts intersections between different files of different datasets
import sys
import itertools
from itertools import combinations, chain

data_path = "../data/"

FILES = [
    'macmorpho-train.mm',
    'macmorpho-dev.mm',
    'macmorpho-test.mm',
    'pt_bosque-ud-train.mm',
    'pt_bosque-ud-dev.mm',
    'pt_bosque-ud-test.mm',
    'pt_gsd-ud-train.mm',
    'pt_gsd-ud-dev.mm',
    'pt_gsd-ud-test.mm',
    'lgtc-train.mm',
    'lgtc-dev.mm',
    'lgtc-test.mm',
]


def findsubsets(s, n):
    return list(map(set, itertools.combinations(s, n)))

def open_file(file_name):
    print(">> Trying to open file...")
    try:
        f = open(file_name, "rt")
    except:
        print(">>>> Unable to open file")
        exit()
    print(">>> File was successfully opened")
    return f.read()

def get_sents(text):
    return [[t.split('_')[0] for t in sent.split(' ')] for sent in text.split('\n')]

def process_file(data_path, file_name, d):
    text = open_file(data_path + file_name)
    sents = get_sents(text)

    for s in sents:
        s = " ".join(s)
        if len(s) == 0:
            continue

        s = s.strip()
        if s in d:
            d[s].add(file_name)
        else:
            d[s] = {file_name}

    return d



# saves all samples on a dict, where the key is the sample and the value are the
# files in which it appears
d = dict()
for file_name in FILES:
    d = process_file(data_path, file_name, d)


print('\n')


intersect_files = dict()

for sent, files in d.items():
    files = list(set(files))
    files.sort()
    files = tuple(files)

    # recovers all datasets from files the sentence appears
    datasets = set()
    for f in files:
        datasets.add(f.split('-')[0])
    datasets = list(datasets)
    datasets.sort()
    datasets = tuple(datasets)

    # only interested in intersection between two or more different datasets
    if(len(datasets) == 1):
        continue

    # generate all pairs from retrieved files
    x = findsubsets(files, 2)
    for e in x:
        # orders and make e hashing possible
        e = list(e)
        e.sort()
        e = tuple(e)

        # counter for each possible pair of files
        if e not in intersect_files:
            intersect_files[e] = 1
        else:
            intersect_files[e] += 1

# recovers values from dict and orders them
l = list(intersect_files.items())
l.sort()

# show results
for files, count in l:
    print(files, count)
