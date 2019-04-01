'''
Text in the corpus needs to be converted to a format that can be interpreted by the machine learning algorithms. 
There are 2 parts of this conversion — Tokenisation and Vectorisation.

Tokenisation is the process of converting the continuous text into a list of words.
The list of words is then converted to a matrix of integers by the process of vectorisation. 
Vectorisation is also called feature extraction.

For text preparation we use the bag of words model which ignores the sequence of the words and only considers word frequencies.
'''

from sklearn.feature_extraction.text import CountVectorizer
import re
import pickle
import numpy as np
import pandas as pd
from text_preprocessing import stop_words

# Creating a vector of word counts - tokenise the text and build a vocabulary of known words.
class Tokenizer:

	@staticmethod
	def get_word_frequencies(corpus_pickle):
		with open(corpus_pickle, 'rb') as pickle_in:
			corpus = pickle.load(pickle_in, encoding='utf8')

		cv = CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
		normalized = cv.fit_transform(corpus)

		with open('../bin/count_vectorizer.pkl', 'wb') as output:
			pickle.dump(cv, output)
			output.close()

		with open('../bin/normalized.pkl', 'wb') as output:
			pickle.dump(normalized, output)
			output.close()

		bag_of_words = cv.transform(corpus)
		sum_words = bag_of_words.sum(axis=0) 
		words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
		words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

		with open('../bin/words_freq.pkl', 'wb') as output:
			pickle.dump(words_freq, output)
			output.close()

		print(words_freq)

def main():
    Tokenizer.get_word_frequencies('../bin/corpus.pkl')

if __name__ == "__main__":
   main()
