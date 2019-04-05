from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import pickle
import os

CORPUS_DIR='bin/train_corpus.pkl'
N_GRAM_RANGE=(1,3)

# Bag Of Words Model: calculate ngram frequencies - tokenise the text and build a vocabulary of tokens.
# It takes into account only the frequency of the words in the vocabulary, not their order or position
class BagOfWords:
	@staticmethod
	def ngram_frequencies_gen(corpus_dir=CORPUS_DIR, ngram_range=N_GRAM_RANGE):
		with open(corpus_dir, 'rb') as pickle_in:
			corpus = pickle.load(pickle_in, encoding='utf8')

		# The below breaks up the words into n-grams of length of N_GRAM_RANGE and puts their counts into a Pandas dataframe 
		# with the n-grams as column names. 
		ngram_bow = CountVectorizer(max_df=0.9, max_features=None, analyzer='word', ngram_range=ngram_range)
		ngram_count_sparse = ngram_bow.fit_transform(corpus) # Here we deteremine the model vocabulary 
		ngram_count = pd.DataFrame(ngram_count_sparse.toarray())
		ngram_count.columns = ngram_bow.get_feature_names()

		# The below turns the n-gram-count dataframe into a Pandas series with the n-grams as indices for ease of working with the counts.
		# The second line can be used to limit the n-grams used to those with a count over a cutoff value.
		sums = ngram_count.sum(axis=0) 
		sums = sums[sums > 0]
		# list of all ngrams from corpora
		ngram_list = list(sums.index.values)

		# Return ngram categories, which is a dictionary with a key of the gram number (1: unigram, 2: bigram, ...) with values being a list
		# of those ngram tokens
		def get_ngrams():
			ngrams = defaultdict(list)
			for ng in ngram_list:
				ng_split = ng.split(" ")
				ngrams[len(ng_split)].append(ng)
			return ngrams

		def base_freq(unigram_counts):
		    freqs = {}
		    for ng in ngram_list:
		        ng_split = ng.split(" ")
		        if len(ng_split) == 1:
		        	freqs[ng] = sums[ng] / unigram_counts
		    return freqs

		ngrams = get_ngrams()
		unigram_freq = base_freq(len(ngrams[1]))
		
		with open('bin/unigram_freq.pkl', 'wb') as output:
			pickle.dump(unigram_freq, output)
			output.close()

		with open('bin/ngrams.pkl', 'wb') as output:
			pickle.dump(ngrams, output)
			output.close()

def main():
	BagOfWords.ngram_frequencies_gen()

if __name__ == "__main__":
   main()
