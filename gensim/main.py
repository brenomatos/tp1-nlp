from gensim.models import Word2Vec
import gensim
from nltk.tokenize import word_tokenize
from itertools import chain
from glob import glob
from nltk.tokenize import word_tokenize
import os,logging


tng_size = 200
tng_window = 5
tng_min_count = 1
tng_workers = 5
tng_iter = 5
tng_sg = 0

def to_lower(input_file,output_file):#formatando o arquivo de validacao para minusculo
    file = open(input_file, 'r')
    lines = [line.lower() for line in file]
    with open(output_file, 'w') as out:
         out.writelines((lines))

def treino_word2vec(tng_size,tng_window,tng_min_count,tng_workers,tng_iter,tng_sg,output_path):
    corpus = gensim.models.word2vec.Text8Corpus('text8', max_sentence_length=10000)
    model = gensim.models.Word2Vec(corpus,size=tng_size, window=tng_window, min_count=tng_min_count, workers=tng_workers, iter=tng_iter,sg=tng_sg)
    model.save(output_path)
    return model

def similaridade(input_file, model, output_file, csv_output):
    #palavra0 palavra1 palavra2 palavra3 eh o modelo que tem no questions-words
    #Palavra1 e palavra2 são as positivas. Palavra0 negativa.
    csv = open(csv_output, "w")
    csv.write("correct_word,correct_word_similarity,top_minus_correct_similarity\n")
    file = open(input_file, 'r')
    lines = [line.lower() for line in file]
    for i in lines:
        word_tokens = word_tokenize(i)
        if(word_tokens[0]!=':'): #para ignorar as linhas com dois pontos
            try:
                similarities = model.wv.most_similar(positive=[word_tokens[1], word_tokens[2]],negative=[word_tokens[0]], topn=50)
                for i in range(0, len(similarities)):
                    if (similarities[i][0] == word_tokens[3]):
                        break
                # A linha abaixo imprime, nessa ordem: a palavra correta, sua similaridade dadas as 3 outras palavras,
                # a diferenca entre a com maior similaridade e a similaridade da palavra correta, seguida pelo indice
                # da palavra correta no top 50
                csv.write(word_tokens[3]+","+str(similarities[i][1])+","+str(similarities[0][1]-similarities[i][1])+","+str(i)+"\n")
                # print(word_tokens[3]+","+str(similarities[i][1])+","+str(similarities[0][1]-similarities[i][1])+","+str(i))
            except Exception as e:
                pass #ignorando as instancias fora do vocabulario
    csv.close()
    file.close()


output_path = "w2v-"+str(tng_size)+"-"+str(tng_window)+"-"+str(tng_min_count)+"-"+str(tng_workers)+"-"+str(tng_iter)+"-"+str(tng_sg)
logging.basicConfig(filename=output_path+"log",format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# model = gensim.models.Word2Vec.load('w2v-200-5-1-5-5-0')
model = treino_word2vec(tng_size,tng_window,tng_min_count,tng_workers,tng_iter,tng_sg,output_path)
similaridade("questions-words.txt", model, output_path, output_path+".csv")

model.wv.accuracy('questions-words.txt')#resulta em estatisticas no log criado
os.remove(output_path+".trainables.syn1neg.npy")#deletando arquivos desnecessarios
os.remove(output_path+".wv.vectors.npy")
