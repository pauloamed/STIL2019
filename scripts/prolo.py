# Converts corpus on mm format to prolos format
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

def create_file(file_name, text):
    try:
        file = open(file_name, "w")
    except:
        print(">>>> Unable to create file")
        exit()

    print(file_name)

    text = text.split('\n')
    for i, sample in enumerate(text):
        index = i+1
        if len(str(index)) < len(str(len(text))):
            index = ("0" * (len(str(len(text))) - len(str(index)))) + str(index)
        for token in sample.split(' '):
            if token == '':
                continue
            token = token.rsplit('_')
            file.write(str(index) + " " + token[0] + " " + token[1])
            file.write("\n")
    file.close()

    print(">>> File was successfully created")


file_name = sys.argv[1]

text = open_file(file_name)
create_file(file_name[:-3] + ".prolo", text)
