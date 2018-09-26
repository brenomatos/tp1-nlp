from gensim.models import Word2Vec
import gensim
from nltk.tokenize import word_tokenize

# f=open('brasil.txt','r')
# corpus = f.read()
# tokenized_text = word_tokenize(corpus)
# print(tokenized_text)


corpus = gensim.models.word2vec.Text8Corpus('brasil.txt', max_sentence_length=10000)
model = gensim.models.Word2Vec(corpus,size=100, window=2, min_count=1, workers=5)

model.save("bin")
context_words_list = ["Rio", "de","Janeiro"]
print(model.predict_output_word(context_words_list, topn=10))
