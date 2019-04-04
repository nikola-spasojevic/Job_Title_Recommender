'''
Text in the corpus needs to be converted into a format that can be interpreted by machine learning algorithms. 
Tokenisation is the process of converting the continuous text into a list of words/phrases.
For text preparation we use the bag of words model which ignores the sequence of the words and only considers word frequencies.
'''

from sklearn.feature_extraction.text import CountVectorizer
from nltk import FreqDist
import re
import pickle
import numpy as np
import pandas as pd
from text_preprocessing import stop_words
from collections import Counter

CORPUS_DIR = '../bin/corpus.pkl'
N_GRAM_RANGE = (1,2)

# Create an instance of the CountVectorizer class.
# Call the fit() function in order to learn a vocabulary from one or more documents.
# Call the transform() function on one or more documents as needed to encode each as a vector.

# Bag Of Words Model: calculate ngram frequencies - tokenise the text and build a vocabulary of tokens.
# It takes into account only the frequency of the words in the language, not their order or position
class BagOfWords:
	@staticmethod
	def ngram_frequencies(corpus_dir, ngram_range):
		with open(corpus_dir, 'rb') as pickle_in:
			corpus = pickle.load(pickle_in, encoding='utf8')

		# Convert a collection of text documents to a matrix of token counts
		# potentially set ngram_range=(1,2) to include bigram vocabulary and frequencies
		cv = CountVectorizer(max_df=0.9, stop_words=stop_words, max_features=None, analyzer='word', ngram_range=ngram_range)
		# Learn the vocabulary dictionary and return term-document matrix.
		cv.fit(corpus) # Fit the Data
		bag_of_words = cv.transform(corpus)
		sum_words = bag_of_words.sum(axis=0) 
		ngram_freq = {word: sum_words[0, idx] for word, idx in cv.vocabulary_.items()}
		
		with open('../bin/ngram_freq.pkl', 'wb') as output:
			pickle.dump(ngram_freq, output)
			output.close()

def main():
    BagOfWords.ngram_frequencies(CORPUS_DIR, ngram_range=N_GRAM_RANGE)

if __name__ == "__main__":
   main()
