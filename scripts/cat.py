# Concat list of texts and removes tags
import sys

def open_file(file_name):
    print(">> Trying to open file...")
    try:
        f = open(file_name, "rt")
    except:
        print(">>>> Unable to open file")
        exit()
    print(">>> File was successfully opened")
    return f.read().split('\n')

def create_file(file_name, samples):
    try:
        file = open(file_name, "w")
    except:
        print(">>>> Unable to create file")
        exit()

    for sample in samples:
        if sample == []:
            continue
        for i in range(len(sample)):
            file.write(sample[i])
            if i < len(sample)-2:
                file.write(" ")
        file.write("\n")
    file.close()

    print(">>> File was successfully created")

def rm_tokens(samples):
    for i in range(len(samples)):
        samples[i] = samples[i].split(" ")
        for ii in range(len(samples[i])):
            samples[i][ii] = samples[i][ii].rsplit('_')[0]
    return samples



num_files = int(sys.argv[1])
samples = [sample for i in range(num_files) for sample in open_file(sys.argv[i+2])]
samples = rm_tokens(samples)
create_file(sys.argv[num_files+2], samples)
