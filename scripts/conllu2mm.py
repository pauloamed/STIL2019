# Converts corpus on conllu format to the macmorpho format
from re import compile
import sys

range_r = compile(r"(\d+)-(\d+)")
num_r = compile(r"(\d+)")


def open_file(file_name):
    print(">> Trying to open file...")
    try:
        f = open(file_name, "rt")
    except:
        print(">>>> Unable to open file")
        exit()
    print(">>> File was successfully opened")
    return f.read()

def split_file(f):
    return f.split('\n\n')

def split_sample(sample):
    return sample.split('\n')

def get_situation(s):
    if(range_r.fullmatch(s)):
        start, end = s.split('-')
        return list(range(int(start), int(end)+1))
    elif(num_r.fullmatch(s)):
        return [int(s)]
    else:
        return []

def extract_from_sample(sample):
    samples = []
    for line in split_sample(sample):
        try:
            line = line.split('\t')
            ret = get_situation(line[0])
            if len(ret) == 1:
                samples.append([line[1], line[3]])
        except:
            pass

    return samples

def create_file(file_name, converted_samples):
    try:
        file = open(file_name, "w")
    except:
        print(">>>> Unable to create file")
        exit()

    for sample in converted_samples:
        if sample == []:
            continue
        for i in range(len(sample)):
            file.write(sample[i][0]+"_"+sample[i][1])
            if i < len(sample)-1:
                file.write(" ")
        file.write("\n")
    file.close()

    print(">>> File was successfully created")


file_dest = sys.argv[1]
processed_file_dest = sys.argv[2]
f = open_file(file_dest)
samples = split_file(f)
samples = [extract_from_sample(sample) for sample in samples]
create_file(processed_file_dest, samples)
