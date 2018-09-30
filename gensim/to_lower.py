from itertools import chain
from glob import glob

def to_lower(input_file,output_file):
    file = open(input_file, 'r')
    lines = [line.lower() for line in file]
    with open(output_file, 'w') as out:
         out.writelines((lines))


# to_lower("questions-words.txt","questions-words.txt")

def resize_corpus(input_file, output_file, percentage):
    with open(input_file, "r") as input:
        lines = input.read().splitlines()
        # lines = [for line in file]
        print(len(lines))
        # for line in lines:
            # print(line+"\n\n\n")

resize_corpus("text8","lixo",50)
