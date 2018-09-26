from gensim.models import Word2Vec
import gensim
from nltk.tokenize import word_tokenize

# f=open('brasil.txt','r')
# corpus = f.read()
# tokenized_text = word_tokenize(corpus)
# print(tokenized_text)


#corpus = gensim.models.word2vec.Text8Corpus('text8', max_sentence_length=10000)
#model = gensim.models.Word2Vec(corpus,size=200, window=5, min_count=1, workers=5, iter=5,sg=0)

#model.save("bin")
model = gensim.models.KeyedVectors.load_word2vec_format("bin")
context_words_list = ["mexico", "mexican","macedonia"]
print(model.predict_output_word(context_words_list, topn=20))



#most_similar_to_given(entity1, entities_list)¶

