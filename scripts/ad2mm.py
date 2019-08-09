# Converts corpus on ad format to the macmorpho format
from re import compile
import sys
import string


def open_file(file_name):
    print(">> Trying to open file...")
    # try:
    f = open(file_name, "rt", encoding="utf-8").read()
    # except:
    #     print(">>>> Unable to open file")
    #     exit()
    print(">>> File was successfully opened")
    return f

def split_file(f):
    return [s.strip() for s in f.split('<s>')]

def create_file(file_name, converted_samples):
    try:
        file = open(file_name, "w")
    except:
        print(">>>> Unable to create file")
        exit()

    for sample in converted_samples:
        if sample == [] or not sample:
            continue
        for i in range(len(sample)):
            file.write(sample[i][0]+"_"+sample[i][1])
            if i < len(sample)-1:
                file.write(" ")
        file.write("\n")
    file.close()

    print(">>> File was successfully created")

def extract_from_sample(sample):
    converted_sample = []
    for line in sample.split('\n'):
        if len(line) > 0 and line[0] == '=':
            i = 0
            while line[i] == '=':
                i += 1
            if all(char in string.punctuation or char in '«»' for char in line[i:]):
                # line = line.replace('»', '>')
                # line = line.replace('«', '<')
                converted_sample.append([line[i:],"punct"])
                continue
            else:
                while i < len(line) and line[i] != ':':
                    i+=1
                inicio = i
                while i < len(line) and line[i] != '\'':
                    i+=1
                fim = i

                if max(inicio, fim) < len(line) and line[inicio] == ':' and line[fim] == '\'':
                    token = line[inicio+1:fim-1]
                    palavra = line.split()[-1]
                    converted_sample.append([palavra, token])
    return converted_sample


file_dest = sys.argv[1]
processed_file_dest = sys.argv[2]
f = open_file(file_dest)
samples = split_file(f)
samples = [extract_from_sample(sample) for sample in samples]
create_file(processed_file_dest, samples)
