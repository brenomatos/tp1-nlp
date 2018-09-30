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
        count = 0
        str_aux=""
        list_final = []
        for line in input:
            for word in line.split():
                str_aux = str_aux+word+" "
                count = count+1
                if(count==10):
                    list_final.append(str_aux)
                    count = 0
                    str_aux=""
    with open(output_file, "w") as output:
        limit = len(list_final)/100*percentage
        count = 0
        for line in range(len(list_final)+1):
            if (count<=limit):
                output.write(list_final[line])
                count = count + 1

# resize_corpus("text8","lixo",75)
