# Converts corpus on conll format to the macmorpho format
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



d = dict()
for file_name in FILES:
    d = process_file(data_path, file_name, d)


print('\n')

counter = 0

intersect_datasets = dict()
intersect_files = dict()

for sent, files in d.items():
    datasets = set()
    subsets = set()
    files = list(set(files))
    files.sort()
    files = tuple(files)
    for f in files:
        datasets.add(f.split('-')[0])
        subsets.add(f.split('-')[-1])

    datasets = list(datasets)
    subsets = list(subsets)

    datasets.sort()
    subsets.sort()

    subsets = tuple(subsets)
    datasets = tuple(datasets)

    if(len(datasets) == 1):
        continue

    if datasets not in intersect_datasets:
        intersect_datasets[datasets] = 1
    else:
        intersect_datasets[datasets] += 1

    x = findsubsets(files, 2)
    for e in x:
        e = list(e)
        e.sort()
        e = tuple(e)
        if e not in intersect_files:
            intersect_files[e] = 1
        else:
            intersect_files[e] += 1


    counter += 1
#     print("{}: Frase é: {}".format(counter, sent))
#     print("Ocorreu em: {}".format(" ".join(files)))
#     print()
# print(counter)

# for datasets, count in intersect_datasets.items():
#     print(datasets, count)
# print()
l = list(intersect_files.items())
l.sort()
for files, count in l:
    print(files, count)
