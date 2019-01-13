# Word2vec
Code regarding the first assignment of Natural Language Processing class @ UFMG. The assignment was to create and test various word2vec models, varying parameters like corpus size, use of skip-gram or CBOW algorithm and context window.

## Getting Started
### Files Needed
Is this assignment, we used the Matt Mahoney's [text8]( http://mattmahoney.net/dc/text8.zip) to train our models. To evaluate them, we used Google's [questions-words.txt](https://code.google.com/archive/p/word2vec/source/default/source).

### Installing
First, clone this repository. Then, You'll need [Gensim](https://radimrehurek.com/gensim/), [NLTK](https://www.nltk.org/) and [Matplotlib](https://matplotlib.org/). You can install those by using pip3 on a terminal:
```bash
pip3 install nltk
pip3 install gensim
pip3 install matplotlib
```

Also, You may need to run this code snippet if it's the first time you use the nltk library
```python
import nltk
nltk.download('punkt')
```

### Running the Code
Open a terminal and run the command:
```bash
python3 main.py
```

## Results
For every result, we generate 3 graphs, like the examples below:

Similarity Boxplot:

![Example Graph: Similarity Boxplot](/results/corpus75-w2v-200-8-1-5-5-1-results/boxplot-corpus100-w2v-200-10-1-5-5-0.jpg "Similarity Boxplot")

Similarity Error Boxplot:

![Example Graph: Error Boxplot](/results/corpus75-w2v-200-8-1-5-5-1-results/boxplot-error-corpus100-w2v-200-10-1-5-5-0.jpg "Error Boxplot")

Similarity Scatterplot:

![Example Graph: Similarity Scatterplot](/results/corpus75-w2v-200-8-1-5-5-1-results/scatter-corpus100-w2v-200-10-1-5-5-0.jpg "Similarity Scatterplot")

Note that every graph's name is structured like:
```bash
Corpus file+"-w2v-"+ size(Dimensionality of the word vectors) + window(Maximum distance between the current and predicted word within a sentence) + min_count(Ignores all words with total frequency lower than this) + workers(number of threads) + iter(number of epochs) + sg(1 = skip-gram; 0 = CBOW)
```

Also, one log file that keeps statistics of hits/misses. Unfortunately, I have not come up with a parsing solution to process this data, but it's not hard to do it by hand.


## Built With
- [Gensim](https://radimrehurek.com/gensim/)

- [NLTK](https://www.nltk.org/),

- [Matplotlib](https://matplotlib.org/)
