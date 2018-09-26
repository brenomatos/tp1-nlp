from itertools import chain
from glob import glob

file = open('questions-words.txt', 'r')

lines = [line.lower() for line in file]
with open('questions-words_lower.txt', 'w') as out:
     out.writelines(sorted(lines))