# Converts corpus on conll format to the macmorpho format
import sys

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
        if s in d:
            d[s].append(file_name)
        else:
            d[s] = [file_name]

    return d



d = dict()
for file_name in FILES:
    d = process_file(data_path, file_name, d)


counter = 0
for sent, files in d.items():
    if(len(files) > 1):
        datasets = set()
        subsets = set()
        for f in files:
            datasets.add(f.split('-')[0])
            subsets.add(f.split('-')[-1])

        if(len(datasets) == 1 or len(subsets) == 1):
            continue
        if(len(datasets) == 2 and 'macmorpho' in datasets and 'lgtc' in datasets):
            counter += 1

        print("Frase é: {}".format(sent))
        print("Ocorreu em: {}".format(" ".join(files)))
        print()
print(counter)
