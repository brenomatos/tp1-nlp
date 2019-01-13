from gensim.models import Word2Vec
import gensim
from nltk.tokenize import word_tokenize
from itertools import chain
from glob import glob
from nltk.tokenize import word_tokenize
import os,logging
from plot import plot
from shutil import copyfile

# This method resizes the corpus to a fraction of it's original size. The
# parameter percentage is used to fix what percentage of the original corpus
# you'll get on a new file (output_file)
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

def to_lower(input_file,output_file):# Formatting the "questions-words.txt" to lowercase
    file = open(input_file, 'r')
    lines = [line.lower() for line in file]
    with open(output_file, 'w') as out:
         out.writelines((lines))

def return_word2vec_model(corpus_file,tng_size,tng_window,tng_min_count,tng_workers,tng_iter,tng_sg,output_path):
    corpus = gensim.models.word2vec.Text8Corpus(corpus_file, max_sentence_length=10000)
    model = gensim.models.Word2Vec(corpus,size=tng_size, window=tng_window, min_count=tng_min_count, workers=tng_workers, iter=tng_iter,sg=tng_sg)
    model.save(output_path)
    return model

def similarity(input_file, model, output_file, csv_output):
    #word0 word1 word2 word3 is the template used in "questions-words.txt"
    #word1 and word2 are "positive". word0 is "negative".
    csv = open(csv_output, "w")
    csv.write("correct_word,correct_word_similarity,top_minus_correct_similarity,similarity_index\n")
    file = open(input_file, 'r')
    lines = [line.lower() for line in file]
    for i in lines:
        word_tokens = word_tokenize(i)
        if(word_tokens[0]!=':'): # ignores the lines with ":"
            try:
                similarities = model.wv.most_similar(positive=[word_tokens[1], word_tokens[2]],negative=[word_tokens[0]], topn=50)
                for i in range(0, len(similarities)):
                    if (similarities[i][0] == word_tokens[3]):
                        break
                # The line below prints, in this order: the correct word, it's similarity given the other 3 words,
                # the diference between the top ranking word's similarity and the similarity of the correct word,
                # followed by the index of the correct word in the top 50 suggested words
                csv.write(word_tokens[3]+","+str(similarities[i][1])+","+str(similarities[0][1]-similarities[i][1])+","+str(i)+"\n")

            except Exception as e:
                pass # Ignore "out of vocabulary" instances
    csv.close()
    file.close()

# the main method preprocesses the questions-words file, trains the model, tests it's accuracy e generates graphs
def main(corpus_file,tng_size,tng_window,tng_min_count,tng_workers,tng_iter,tng_sg):
    to_lower("questions-words.txt","questions-words.txt")# altering the file to lowercase, just to be sure
    output_path = corpus_file +"-w2v-"+str(tng_size)+"-"+str(tng_window)+"-"+str(tng_min_count)+"-"+str(tng_workers)+"-"+str(tng_iter)+"-"+str(tng_sg)
    logging.basicConfig(filename="logTotal.log",format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, filemode="w")

    model = return_word2vec_model(corpus_file,tng_size,tng_window,tng_min_count,tng_workers,tng_iter,tng_sg,output_path)
    similarity("questions-words.txt", model, output_path, output_path+".csv")

    model.wv.accuracy('questions-words.txt')# creates statistics in the log file
    os.remove(output_path+".trainables.syn1neg.npy")# deletes unnecessary files
    os.remove(output_path+".wv.vectors.npy")
    plot(output_path+".csv")

tng_size = 200
# tng_window = 5
tng_min_count = 1
tng_workers = 5
tng_iter = 5
# tng_sg = 0

current_directory = os.getcwd()
copyfile(current_directory+"/text8", current_directory+"/corpus100")# keeps the same pattern
resize_corpus("text8","corpus75",75)
resize_corpus("text8","corpus50",50)

for corpus_file in range(50,125,25):
    for tng_window in range(2, 12, 2):
        for tng_sg in range(0,2):
            print(str(corpus_file)+"-"+str(tng_window)+"-"+str(tng_sg))
            main("corpus"+str(corpus_file),tng_size,tng_window,tng_min_count,tng_workers,tng_iter,tng_sg)
