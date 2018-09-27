from itertools import chain
from glob import glob

def to_lower(input_file,output_file):
    file = open(input_file, 'r')
    lines = [line.lower() for line in file]
    with open(output_file, 'w') as out:
         out.writelines((lines))
