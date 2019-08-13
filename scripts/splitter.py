# file splitter, given weights for each file
# assumes that consecutive elements are separated using \n
import sys

def open_file(file_name):
    print(">> Trying to open file...")
    try:
        f = open(file_name, "rt")
    except:
        print(">>>> Unable to open file")
        exit()
    print(">>> File was successfully opened")
    return f.read()

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


file_name = sys.argv[1]
file1_dest = sys.argv[2]
file2_dest = sys.argv[3]
file1_weight = float(sys.argv[4])

if file1_weight < 0 or file1_weight > 1:
    exit()

f = open_file(file_name)
samples = f.split('\n')

total_words = sum([len(sample) for sample in samples])
file1_words = total_words * file1_weight

cont = 0
i = 0
while i < len(samples):
    cont += len(samples[i])
    if cont >= file1_words:
        break
    i+=1

file1_samples = samples[:i]
file2_samples = samples[i:]

create_file(file1_dest, file1_samples)
create_file(file2_dest, file2_samples)

# create_file(file_name, converted_samples)
