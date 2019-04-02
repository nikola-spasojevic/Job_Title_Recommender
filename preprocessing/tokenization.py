'''
Text in the corpus needs to be converted into a format that can be interpreted by machine learning algorithms. 

Tokenisation is the process of converting the continuous text into a list of words/phrases.

For text preparation we use the bag of words model which ignores the sequence of the words and only considers word frequencies.
'''

from sklearn.feature_extraction.text import CountVectorizer
import re
import pickle
import numpy as np
import pandas as pd
from text_preprocessing import stop_words

# unigram/bigram/trigram counts - tokenise the text and build a vocabulary of known words.
class Tokenizer:

	@staticmethod
	def get_word_frequencies(corpus_path):
		with open(corpus_path, 'rb') as pickle_in:
			corpus = pickle.load(pickle_in, encoding='utf8')

		# Convert a collection of text documents to a matrix of token counts
		# potentially set ngram_range=(1,2) to include bigram frequencies and lexicon
		cv = CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
		# Learn the vocabulary dictionary and return term-document matrix.
		X = cv.fit_transform(corpus)
		
		bag_of_words = cv.transform(corpus)
		print(bag_of_words)
		sum_words = bag_of_words.sum(axis=0) 
		word_freq = {word: sum_words[0, idx] for word, idx in cv.vocabulary_.items()}

		# with open('../bin/lexicon.pkl', 'wb') as output:
		# 	pickle.dump(lexicon, output)
		# 	output.close()

		with open('../bin/word_freq.pkl', 'wb') as output:
			pickle.dump(word_freq, output)
			output.close()

def main():
    Tokenizer.get_word_frequencies('../bin/corpus.pkl')

if __name__ == "__main__":
   main()
