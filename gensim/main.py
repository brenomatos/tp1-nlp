from gensim.models import Word2Vec
import gensim
from nltk.tokenize import word_tokenize

corpus = gensim.models.word2vec.Text8Corpus('text8', max_sentence_length=10000)
model = gensim.models.Word2Vec(corpus,size=200, window=5, min_count=1, workers=5, iter=5,sg=0)

model.save("bin")
context_words_list = ["mexico", "mexican","macedonia"]
print(model.predict_output_word(context_words_list, topn=20))

model = gensim.models.Word2Vec.load('bin')
print(model.wv.accuracy('questions-words_lower.txt'))
similarities = model.wv.most_similar(positive=['mexico', 'mexican', "macedonia"], topn=20)
print(similarities)

# esse aqui usa o mesmo questions-words que vem no codigo da google
# https://rare-technologies.com/word2vec-tutorial/
# pip install "h5py==2.8.0rc1"
